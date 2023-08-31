import argparse
import glob
import json
import logging
import os
from datetime import datetime
from uuid import uuid4 as uuid

import numpy as np

with open("./config.json") as f:
    config = json.load(f)
    ACTIVE_GPUS = config["backend"]["activeGpus"]

assert (
    len(ACTIVE_GPUS.split(",")) >= 1
), "Need at least 1 active GPUs set, see serve_diffusion.py"

os.environ["CUDA_VISIBLE_DEVICES"] = ACTIVE_GPUS

import ray
import torch
from backend_config import BackendConfig, CompvisConfig, DiffusersConfig
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.models import ControlNetModel
from logging_util import create_file_logger
from omegaconf import OmegaConf
from pipeline_controlnet import Control, StableDiffusionControlNetPipeline
from pytorch_lightning import seed_everything
from ray import serve
from starlette.requests import Request

with open("../config.json") as f:
    config = json.load(f)
    RAY_BACKEND_HOST = config["backend"]["rayHostAddress"]
    RAY_BACKEND_PORT = int(config["backend"]["rayHostPort"])


def serve_diffusion_model(args, host: str, port: int, logdir: str, backend: str = "diffusers"):
    print("***********************************************************")
    print(f"Serving diffusion model at: {host}:{port}")
    print("***********************************************************")

    if backend == "diffusers":
        config = DiffusersConfig(
            config_name="default",
            repo_id="stabilityai/stable-diffusion-2-base",
            torch_dtype=torch.float16,
            revision="fp16",
        )
    elif backend == "compvis":
        config = CompvisConfig(
            config_name="default",
            omega_config=OmegaConf.load(
                f"/raid/{os.getlogin()}/diffusion-exploration/backend/stable-diffusion/configs/stable-diffusion/v2-inference.yaml"
            ),
            model_ckpt=f"/raid/{os.getlogin()}/diffusion-exploration/backend/stable-diffusion/checkpoints/model.ckpt",
        )
    else:
        raise NotImplementedError()

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dreams")
    os.makedirs(outdir, exist_ok=True)

    serve.start(http_options={"host": host, "port": port})
    generate_handle = DiffusionDeployment.deploy(config=config, outdir=outdir, logdir=logdir)
    poll_handle = PollDeployment.deploy(outdir=outdir, logdir=logdir)
    # finetune_handle = FinetuneDeployment.deploy()

    return (generate_handle, poll_handle)


# Return the latest step else return -1 if doesn't exist
def get_latest_step(job_dir: str, ext: str):
    if os.path.exists(job_dir):
        # NOTE lambda is very ad hoc and specific to our naming convention
        # e.g. [STEP].png and intermediate_[STEP].pt
        steps = list(
            map(
                lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]),
                glob.glob(f"{job_dir}/*.{ext}"),
            )
        )
        if len(steps) > 0:
            latest_step = max(steps)
        else:
            latest_step = -1
    else:
        latest_step = -1

    return latest_step


# Deployment to handle poll requests
@serve.deployment(route_prefix="/poll", num_replicas=1)
class PollDeployment:
    def __init__(self, outdir: str, logdir: str):
        self.logger = create_file_logger(
            name=__name__,
            filename=os.path.join(logdir, f"backend_poll_{uuid()}.log"),
            level=logging.INFO,
        )

        self.logger.info("Creating a PollDeployment...")
        self.outdir = outdir

    async def __call__(self, request: Request):
        return await self.handle_request(request)

    async def handle_request(self, request: Request):
        self.logger.info("(Charlotte) received POLL request...")
        data = await request.json()

        # Parse the request body

        job_ids = data["job_ids"]
        job_dirs = [os.path.join(self.outdir, job_id) for job_id in job_ids]
        latest_steps = [get_latest_step(job_dir, "png") for job_dir in job_dirs]

        return {"job_ids": job_ids, "latest_steps": latest_steps}


# @serve.deployment(
#     route_prefix="/finetune",
#     num_replicas=len(ACTIVE_GPUS.split(",")) - 1,
#     ray_actor_options={"num_gpus": 1},
# )
class FinetuneDeployment:
    def __init__(self):
        pass

    async def handle_request(self, request: Request):
        print("Received finetune request... \n\n\n")
        text_prompt = request.query_params["text_prompt"]
        instance_dir = request.query_params["instance_dir"]
        output_dir = request.query_params["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        model_name = "/raid/sarukkai/stable-diffusion-2-1-base"
        lr = 4e-4
        lr_warmup_steps = 25

        command_args = [
            f"--pretrained_model_name_or_path={model_name}",
            f"--instance_data_dir={instance_dir}",
            f"--output_dir={output_dir}",
            "--train_text_encoder",
            "--resolution=512",
            "--train_batch_size=4",
            "--gradient_accumulation_steps=1",
            f"--learning_rate_ti={lr}",
            "--color_jitter",
            "--lr_scheduler=constant",
            f"--lr_warmup_steps={lr_warmup_steps}",
            "--mixed_precision=fp16",
            "--max_train_steps=250",
            f"--placeholder_token=<krk>",
            f"--instance_prompt=<krk> {text_prompt}",
            "--learnable_property=object",
            "--initializer_token=a",
            "--save_steps=50",
            "--unfreeze_lora_step=1000",
        ]
        from finetune import main

        main(command_args)
        return {}

    async def __call__(self, request: Request):
        return await self.handle_request(request)


# Deployment to handle generate requests
@serve.deployment(
    route_prefix="/generate",
    num_replicas=len(ACTIVE_GPUS.split(",")),
    ray_actor_options={"num_gpus": 1},
)
class DiffusionDeployment:
    def __init__(self, config: BackendConfig, outdir: str, logdir: str):
        self.logger = create_file_logger(
            name=__name__,
            filename=os.path.join(logdir, f"backend_diffusion_{uuid()}.log"),
            level=logging.INFO,
        )

        self.logger.info(
            f"Creating a DiffusionDeployment, available gpus: {ray.get_gpu_ids()}..."
        )

        self.config = config
        self.outdir = outdir

        if isinstance(config, DiffusersConfig):
            self.backend_name = "diffusers"
            controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=[controlnet_canny], 
            )
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
            )
            # TODO: refactor this
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.pipe = self.pipe.to(self.device)
            # self.pipe.setup()
            self.logger.info("Using DIFFUSERS backend")
        elif isinstance(config, CompvisConfig):
            raise NotImplementedError
        else:
            raise NotImplementedError

    async def __call__(self, request: Request):
        return await self.handle_request(request)

    async def handle_request(self, request: Request):
        self.logger.info("(Charlotte) received GENERATE request...")
        data = await request.json()

        self.logger.info(
            f"Processing diffusion request for: job={data['job_id']}, start={data['start_step']}, end={data['end_step']}"
        )
        # print(data)
        # print(f"running diffusion from {data['start_step']} to {data['end_step']}")
        # print()

        # Check if steps already exist
        job_dir = os.path.join(self.outdir, data["job_id"])

        # TODO fix this, logic currently only works for batch size 1
        # TODO also i don't think this even works lol
        paths = []
        if data["name"] == "similar":
            self.logger.info("Generating similar images...")
            paths = self.generate_similar(
                original_job_id=data["original_job_id"],
                job_id=data["job_id"],
                prompt=data["prompt"],
                num_inference_steps=data["total_steps"],
                scale=data["guidance_scale"],
                seed=data["seed"],
            )
        elif data["name"] == "collage":
            cac_strengths = [layer["cacStrength"] for layer in data["collage_layers"]]
            cac_negative_strengths = [layer["negativeStrength"] for layer in data["collage_layers"]]
            canny_strengths = [layer["cannyStrength"] for layer in data["collage_layers"]]
            (
                composite_image,
                mask_layers,
                attention_mod,
                collage_prompt,
            ) = self.pipe.preprocess_layers(
                data["collage_layers"],
                cac_strengths,
                cac_negative_strengths,
                data["prompt"],
            )
            
            # Note:
            # Default is 0.8, but eventually set this to a value between 0 and 1 
            # passed in from the frontend
            noise_strengths = [layer['noiseStrength'] for layer in data['collage_layers']]
            img2img_strength = max(noise_strengths)
            generator = torch.Generator(self.device).manual_seed(data["seed"]) if data['seed'] else None

            canny_mask = self.pipe.generate_control_mask(mask_layers, canny_strengths)
            noise_mask = self.pipe.generate_mask(mask_layers, img2img_strength, noise_strengths)

            image = self.pipe.collage(
                prompt=collage_prompt + ", best quality, 8k, highly detailed",
                negative_prompt='blurry image, worst quality, low quality, collage',
                image=composite_image,
                num_inference_steps=data["total_steps"],
                mask_image=noise_mask,
                strength=img2img_strength, 
                guidance_scale=data['guidance_scale'],
                attention_mod=attention_mod,
                controlnet_conditioning_scale=[canny_mask],
                generator=generator,
                controls=[Control.CANNY],
            ).images[0]

            job_outdir = os.path.join(self.outdir, data["job_id"].split("_")[0])
            os.makedirs(job_outdir, exist_ok=True)
            image.save(os.path.join(job_outdir, f"{data['total_steps']}.png"))
        elif data["name"] == "collage-edit":
            print("Received collage edit request")
            assert data['collage_src'] is not None
            # TODO: do I need to change these?
            cac_strengths = [layer["cacStrength"] for layer in data["collage_layers"]]
            cac_negative_strengths = [layer["negativeStrength"] for layer in data["collage_layers"]]
            (
                composite_image,
                mask_layers,
                attention_mod,
                collage_prompt,
            ) = self.pipe.preprocess_layers(
                data["collage_layers"],
                cac_strengths,
                cac_negative_strengths,
                data["prompt"],
                collage_src=data['collage_src'], # TODO: check
            )
            
            # Note:
            # Default is 0.8, but eventually set this to a value between 0 and 1 
            # passed in from the frontend
            noise_strengths = [layer["noiseStrength"] for layer in data["collage_layers"]]
            img2img_strength = max(noise_strengths)
            print(f"{img2img_strength=}, {noise_strengths=}")

            seed_everything(data["seed"])
            print("Collage prompt: ", collage_prompt)
            print(f"Edit composite image shape: {np.array(composite_image).shape}")
            image = self.pipe.collage(
                prompt=collage_prompt,
                image=composite_image,
                num_inference_steps=data["total_steps"],
                mask_image=self.pipe.generate_mask(mask_layers, img2img_strength, noise_strengths),
                strength=img2img_strength,  # img2img strength
                guidance_scale=data["guidance_scale"],
                attention_mod=attention_mod,
            ).images[0]

            job_outdir = os.path.join(self.outdir, data["job_id"].split("_")[0])
            os.makedirs(job_outdir, exist_ok=True)
            image.save(os.path.join(job_outdir, f"{data['total_steps']}.png"))
        else:
            if data["end_step"] > get_latest_step(job_dir, "png"):
                paths = self.run_diffusion(
                    job_id=data["job_id"],
                    prompt=data["prompt"],
                    start_step=max(
                        data["start_step"], get_latest_step(job_dir, "pt") + 1
                    ),  # TODO check if +1 breaks compvis backend
                    end_step=data["end_step"],
                    num_inference_steps=data["total_steps"],
                    scale=data["guidance_scale"],
                    seed=data["seed"],
                    viz_params=data["viz_params"],
                )

        result = {
            "job_id": data["job_id"],
            "paths": paths,
            "vizParams": data["viz_params"],
        }

        self.logger.info(f"Returning response: {result}")
        return result

    # TODO check if start_step exists
    # TODO save intermediates
    def run_diffusion(
        self,
        job_id,
        start_step,
        end_step,
        prompt="a photo of a person",
        num_inference_steps=50,
        scale=7.5,
        seed=42,
        viz_params=False,
    ):
        seed_everything(seed)

        job_outdir = os.path.join(self.outdir, job_id.split("_")[0])
        os.makedirs(job_outdir, exist_ok=True)

        paths = None  # TODO maybe remove returning paths??

        if self.backend_name == "diffusers":
            self.pipe.partial(
                prompt=prompt,
                start_step=start_step,
                end_step=end_step,
                num_inference_steps=num_inference_steps,
                job_outdir=job_outdir,
            )
        elif self.backend_name == "compvis":
            paths = self.pipe.partial(
                job_id=job_id,
                prompt=prompt,
                start_step=start_step,
                end_step=end_step,
                ddim_steps=num_inference_steps,
                scale=scale,
                job_outdir=job_outdir,
            )
        else:
            raise NotImplementedError()

        return paths

    def generate_similar(
        self,
        original_job_id: str,
        job_id: str,
        prompt: str,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        n_samples: int = 1,
        scale: float = 5.0,
        strength: float = 0.5,
        seed: int = 42,
    ):
        seed_everything(seed)
        job_outdir = os.path.join(self.outdir, job_id.split("_")[0])
        os.makedirs(job_outdir, exist_ok=True)

        if self.backend_name == "compvis":
            self.model.generate_similar(
                original_job_id=original_job_id,
                latest_step=get_latest_step(
                    os.path.join(self.outdir, original_job_id.split("_")[0])
                ),
                prompt=prompt,
                n_samples=n_samples,
                strength=strength,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                scale=scale,
                seed=seed,
                job_outdir=job_outdir,
            )
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start server.")
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--external_deployment", action="store_true")
    parser.add_argument("--port", default=RAY_BACKEND_PORT, type=int)
    args = parser.parse_args()
    port = args.port
    host = "0.0.0.0" if args.external_deployment else "127.0.0.1"

    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    LOGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs", now)
    handle = serve_diffusion_model(args, host=host, port=port, logdir=LOGDIR)

    while True:
        continue
