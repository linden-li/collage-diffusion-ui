# Simple Flask app with a single endpoint for diffusion exploration
from layers import ImageLayer
import numpy as np
import logging
import os
import shutil
import json
import hashlib
import time
from pytz import timezone
from datetime import datetime
from urllib.request import urlopen
from utils.masking import generate_mask, apply_mask

now = datetime.now(timezone("US/Pacific")).strftime("%Y_%m_%d-%H_%M_%S")
LOGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs", now)
# if os.path.exists(LOGDIR):
#     shutil.rmtree(LOGDIR)

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

import argparse
import uvicorn
import pydantic
from typing import Dict, Any
import requests
from starlette.requests import Request
from multiprocessing import Process
from time import sleep
from dotenv import load_dotenv

import ray
from fastapi import FastAPI, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import base64

from data_models import (
    GenerateRequest,
    GenerateRequestBackend,
    GenerateResponse,
    PollRequest,
    PollResponse,
    DirectiveRequest,
    DirectiveResponse,
    CollageRequest,
    CollageEditRequest,
    CollageResponse,
    CollageStatePushResponse,
    FinetuneRequest,
    FinetuneResponse,
    CollageState,
    Layer,
    TokenData,
)
from job_manager import CreateJobManager
from job import Job, DiffusionParams
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from scheduler import run_scheduler
from logging_util import create_file_logger

# from config import RAY_BACKEND_HOST, RAY_BACKEND_PORT, FASTAPI_PORT
from utils.gcloud_utils import upload_to_bucket

with open("./config.json") as f:
    config = json.load(f)
    RAY_BACKEND_HOST = config["backend"]["rayHostAddress"]
    RAY_BACKEND_PORT = int(config["backend"]["rayHostPort"])
    FASTAPI_HOST = config["backend"]["webserverHost"]
    FASTAPI_PORT = int(config["backend"]["webserverPort"])

logger = create_file_logger(
    name=__name__,
    filename=os.path.join(LOGDIR, "api.log"),
    level=logging.DEBUG,
)

IMAGE_DIR_NAME = "dreams"
base_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(base_dir, IMAGE_DIR_NAME)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# TODO: move to setup
collage_json_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "collage_jsons"
)
if not os.path.exists(collage_json_dir):
    os.makedirs(collage_json_dir)


backend = None

api = FastAPI()

# Add middleware to allow CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount directory to statically serve files
api.mount(f"/{IMAGE_DIR_NAME}", StaticFiles(directory="./dreams"), name=IMAGE_DIR_NAME)

# Interprocess synchronization
manager = CreateJobManager()
priority_queue = manager.PriorityQueue()
directives = manager.dict()


@ray.remote
def post_async(request: GenerateRequestBackend):
    """
    Sends an async request to the ray backend.
    """
    return requests.post(
        f"http://{RAY_BACKEND_HOST}:{RAY_BACKEND_PORT}/generate", data=request.json()
    ).json()


def dump_collage_json(state: CollageState):
    collage_json = os.path.join(collage_json_dir, f"{state.collage_id}.json")
    with open(collage_json, "w") as f:
        collage_data = state.json()
        f.write(collage_data)


def read_collage_json(collage_id: str):
    collage_json = os.path.join(collage_json_dir, f"{collage_id}.json")
    if not os.path.exists(collage_json):
        return CollageState(
            collage_id=collage_id,
            collage_prompt="",
            num_inference_steps=50,
            guidance_scale=7,
            seed=42,
            num_frames=5,
            layers=[],
            input_token_data=[TokenData(data=None, layerKey=None, id="start")],
            layer_to_index_map={},
        )

    with open(collage_json, "r") as f:
        data = json.load(f)
        logger.info(f"({type(data)}, json data: {data}")
        collage_data = data
    return collage_data


@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    print(exc_str)
    body = await request.body()
    print(body)
    # or logger.error(f'{exc}')
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=status.HTTP_406_NOT_ACCEPTABLE)


@api.on_event("startup")
async def startup_event():
    """
    On startup, perform the following events:
        (1) Start the scheduler process, which contains a priority queue
        (2) Serve the diffusion model by starting up the Ray deployment
    """
    parser = argparse.ArgumentParser(description="Start server.")
    parser.add_argument("--backend", default="local")
    args = parser.parse_args()

    # Spawn scheduler process
    # NOTE: this creates a separate ray instance...
    # moving this after serve_diffusion_model() causes ray task to deadlock...
    scheduler_process = Process(
        target=run_scheduler,
        args=(priority_queue, directives, 0.02, LOGDIR),
        daemon=True,
    )
    scheduler_process.start()

    if args.backend == "local":
        logger.info("Serving backend on local machine..")

        # Clear out image dir
        logger.info(f"Clearing out {image_dir}")
        shutil.rmtree(image_dir)
        os.mkdir(image_dir)

        # # Start ray service that hosts diffusion
        from serve_diffusion import serve_diffusion_model

        (diffusion_handle, polling_handle) = serve_diffusion_model(
            None, host=RAY_BACKEND_HOST, port=RAY_BACKEND_PORT, logdir=LOGDIR
        )
    elif args.backend == "none":
        pass
    elif args.backend == "together":
        raise NotImplementedError
    else:
        raise Exception("Backend not recognized.")

    backend = args.backend


# Shutdowns the backend server (ray serve instances)
@api.on_event("shutdown")
async def shutdown_event():
    if backend == "local":
        from ray import serve

        logger.info("Shutting down ray serve instances.")
        serve.shutdown()


@api.post("/upload_image")
async def upload_image(file: UploadFile):
    """
    Endpoint: POST /upload_image

    /upload_image is an endpoint that takes in an image file and uploads it
    to Google Cloud Storage, returning the URL to the image.
    """
    print("Got request")
    image_data = await file.read()
    # Compute md5 hash of image
    image_id = hashlib.md5(image_data).hexdigest()
    url = upload_to_bucket(image_data, image_id)
    # add_collage_layer(collage_id, image_id)  # TODO: figure out if we want to eagerly save collage data on upload?
    return {"url": url}


# The sample endpoint takes in a a set of diffusion and visualization
# parameters and writes images to disk.
@api.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Endpoint: POST /generate

    /generate is an async endpoint that adds a job to a priority queue and
    returns a job id.
    """

    # Create job and add it to priority queue
    params = DiffusionParams(
        prompt=request.prompt, guidance_scale=request.guidanceScale, seed=request.seed
    )

    new_job = Job(
        num_inference_steps=request.numInferenceSteps,
        breadth_first=False,  # request.breadthFirst,  # TODO potentially re-enable this
        diffusion_params=params,
        viz_params=request.vizParams,
        batch_size=max(
            2, request.numInferenceSteps // 10
        ),  # TODO figure out why this breaks for n < 2...
        # batch_size=2,
    )

    logger.info(
        f"(api) received /generate request, adding job: {new_job} ({type(new_job)})to priority queue"
    )
    logger.info(
        f"priority_queue: {priority_queue.get_attribute('queue')}, {priority_queue.qsize()}"
    )

    if request.prioritize:
        new_job.priority = -1

    priority_queue.put((new_job.priority, new_job))

    return GenerateResponse(job_id=new_job.job_id, request=request)


@api.post("/generate_similar", response_model=GenerateResponse)
def generate_similar(request: GenerateRequest) -> GenerateResponse:
    """
    Endpoint: POST /generate_similar

    /generate_similar is an async endpoint that adds a job to a priority queue and
    returns a job id.
    """

    # Create job and add it to priority queue
    params = DiffusionParams(
        prompt=request.prompt, guidance_scale=request.guidanceScale, seed=request.seed
    )

    new_job = Job(
        num_inference_steps=request.numInferenceSteps,
        breadth_first=False,
        diffusion_params=params,
        viz_params=request.vizParams,
        original_job_id=request.originalJobId,
        name="similar",
    )

    logger.info(
        f"(api) received /generate_similar request, adding job: {new_job} ({type(new_job)})to priority queue"
    )
    logger.info(
        f"priority_queue: {priority_queue.get_attribute('queue')}, {priority_queue.qsize()}"
    )

    if request.prioritize:
        new_job.priority = -1

    priority_queue.put((new_job.priority, new_job))

    return GenerateResponse(job_id=new_job.job_id, request=request)


@api.post("/directive", response_model=DirectiveResponse)
def directive(request: DirectiveRequest) -> DirectiveResponse:
    """
    Endpoint: POST /directive

    /priotize is an async endpoint that adds a directive to the directives map
    """
    # Prioritize Request
    logger.info(
        f"(api) received DIRECTIVE request: {request.directive} for {request.job_id}"
    )
    directives[request.job_id] = request.directive
    return DirectiveResponse(job_id=request.job_id, directive=request.directive)


@api.post("/generate_blocking", response_model=GenerateResponse)
def generate_blocking(request: GenerateRequest) -> GenerateResponse:
    """
    Endpoint: POST /generate_blocking

    Synchronous endpoint (largely for testing purposes) that will
    submit a job request and wait until it's done.
    """
    logger.info(f"(api) received GENERATE request...\n{request}")

    # Create job and add it to priority queue
    params = DiffusionParams(
        prompt=request.prompt, guidance_scale=request.guidanceScale, seed=request.seed
    )
    job = Job(
        num_inference_steps=request.numInferenceSteps,
        breadth_first=request.breadthFirst,
        diffusion_params=params,
        viz_params=request.vizParams,
        batch_size=1,
    )
    priority_queue.put((job.priority, job))

    # Testing that the scheduler pulls work off the priority queue
    while priority_queue.qsize() > 0:
        sleep(1)
        logger.info("(generate_blocking) waiting...")
        continue

    return GenerateResponse(job_id=job.job_id, request=request)

    # return GenerateResponse(job_id=job.job_id, request=request, data=resp)


@api.post("/poll", response_model=PollResponse)
def poll(request: PollRequest) -> PollResponse:
    """
    Endpoint: POST /poll

    Given a list of job_ids, returns the latest steps that each have completed.
    """
    logger.info(f"(api) received POLL request: {request}...")

    resp = requests.post(
        f"http://{RAY_BACKEND_HOST}:{RAY_BACKEND_PORT}/poll", data=request.json()
    ).json()

    return PollResponse(job_ids=request.job_ids, latest_steps=resp["latest_steps"])


@api.post("/collage", response_model=CollageResponse)
def collage(request: CollageRequest):
    """
    Endpoint: POST /collage

    /collage is an endpoint that adds a job to a priority queue and
    returns a job id.
    """

    params = DiffusionParams(
        prompt=request.prompt, guidance_scale=request.guidanceScale, seed=request.seed
    )

    new_job = Job(
        num_inference_steps=request.numInferenceSteps,
        breadth_first=False,
        diffusion_params=params,
        viz_params=False,
        name="collage",
        collage_layers=request.layers,
    )

    logger.info(
        f"(api) received /collage request, adding job: {new_job} ({type(new_job)})to priority queue"
    )
    logger.info(
        f"priority_queue: {priority_queue.get_attribute('queue')}, {priority_queue.qsize()}"
    )

    # TODO: fix this...
    new_job.priority = -1

    priority_queue.put((new_job.priority, new_job))

    return CollageResponse(job_id=new_job.job_id)


@api.post("/collage_edit", response_model=CollageResponse)
def collage_edit(request: CollageEditRequest):
    """
    Endpoint: POST /collage

    /collage is an endpoint that adds a job to a priority queue and
    returns a job id.
    """

    params = DiffusionParams(
        prompt=request.prompt, guidance_scale=request.guidanceScale, seed=request.seed
    )

    new_job = Job(
        num_inference_steps=request.numInferenceSteps,
        breadth_first=False,
        diffusion_params=params,
        viz_params=False,
        name="collage-edit",
        collage_layers=request.layers,
        collage_src=request.collageSrc,
    )

    logger.info(
        f"(api) received /collage request, adding job: {new_job} ({type(new_job)})to priority queue"
    )
    logger.info(
        f"priority_queue: {priority_queue.get_attribute('queue')}, {priority_queue.qsize()}"
    )

    # TODO: fix this...
    new_job.priority = -1

    priority_queue.put((new_job.priority, new_job))

    return CollageResponse(job_id=new_job.job_id)


@api.post("/push_collage_state", response_model=CollageStatePushResponse)
def push_collage_state(request: CollageState):
    """
    Endpoint: POST /collage_state

    /collage_state is an endpoint that updates the JSON data for a particular collage
    """
    logger.info(
        f"(api) received /push_collage_state request, dumping collage data for {request.collage_id}: {request.layers}"
    )
    dump_collage_json(request)
    return CollageStatePushResponse(collage_id=request.collage_id)


@api.get("/get_collage_state/{collage_id}", response_model=CollageState)
def get_collage_state(collage_id: str) -> CollageState:
    """
    Endpoint: GET /collage_state/{collage_id}
    Given a collage_id, returns collage state.
    """
    logger.info(f"(api) received /get_collage_state request for job_id={collage_id}...")
    state = read_collage_json(collage_id)

    return state


@api.post("/finetune", response_model=FinetuneResponse)
def finetune(request: FinetuneRequest):
    print("Got finetune request")
    layers, layer_index = request.layers, request.layerIdx
    prompt_str = layers[layer_index].textPrompt.replace(" ", "_")
    instance_dir = (
        f"/raid/{os.getlogin()}/diffusion-exploration/ft/asset_library/{prompt_str}"
    )
    os.makedirs(instance_dir, exist_ok=True)
    output_dir = (
        f"/raid/{os.getlogin()}/diffusion-exploration/ft/embeddings/{prompt_str}"
    )
    # The output directory exists, so just return it
    if os.path.exists(output_dir):
        return FinetuneResponse(embedding_path=output_dir)
    layer_list = []
    for layer in layers:
        layer_rgba = Image.open(urlopen(layer.originalImgUrl)).convert("RGBA")
        mask = generate_mask(layer.polygon, layer_rgba.width, layer_rgba.height)
        layer_rgba = apply_mask(layer_rgba, mask)
        # print(f"Word embedding is {layer['wordEmbedding']}")
        image_layer = ImageLayer(
            rgba=layer_rgba,
            image_str=layer.textPrompt,
            pos=(
                int(512 * layer.transform.position.x),
                int(512 * layer.transform.position.y),
            ),
            scale=512 * layer.transform.scale,
            rotation=layer.transform.rotation,
            ftc=layer.wordEmbedding if layer.wordEmbedding is not None else None,
        )

        layer_list.append(image_layer)

    layer_composite, mask_layers = ImageLayer.add_layers(layer_list[: layer_index + 1])
    layer_composite.putalpha(255)
    new_mask = mask_layers[layer_index].copy()
    new_mask = Image.fromarray((new_mask * 255).astype(np.uint8), "L")
    composite_np = np.array(layer_composite)
    composite_np[:, :, 3] = mask_layers[layer_index] * 255
    new_image = Image.fromarray(composite_np, "RGBA")
    new_image.resize((512, 512))
    # Get rid of spaces in the filename
    batch_size = 4
    for i in range(batch_size):
        new_image.save(f"{instance_dir}/{i}.png")

    resp = requests.get(
        f"http://{RAY_BACKEND_HOST}:{RAY_BACKEND_PORT}/finetune",
        params={
            "text_prompt": layer_list[layer_index].image_str,
            "instance_dir": instance_dir,
            "output_dir": output_dir,
        },
    ).json()
    # return the embedding path
    return FinetuneResponse(embedding_path=output_dir)


@api.get("/test")
def test():
    """
    Endpoint: GET /test

    Just for testing fastapi
    """
    logger.info("Received test post request!")
    return True


if __name__ == "__main__":
    uvicorn.run(api, host='0.0.0.0', port=FASTAPI_PORT)
