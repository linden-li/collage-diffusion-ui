import os
import requests
import time
import threading
import traceback
import asyncio
import aiohttp
import json
from threading import Thread
from time import sleep
from queue import PriorityQueue, Empty
import logging

import ray

# from config import RAY_BACKEND_HOST, RAY_BACKEND_PORT
from data_models import GenerateRequestBackend
from job import Job, DiffusionParams
from logging_util import create_file_logger

with open("./config.json") as f:
    config = json.load(f)
    RAY_BACKEND_HOST = config["backend"]["rayHostAddress"]
    RAY_BACKEND_PORT = int(config["backend"]["rayHostPort"])


@ray.remote
class AsyncSubmitter:
    def __init__(self, logdir: str):
        self.logger = create_file_logger(
            name=__name__,
            filename=os.path.join(logdir, "submitter.log"),
            level=logging.DEBUG,
        )

    async def post_async(self, request: GenerateRequestBackend):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{RAY_BACKEND_HOST}:{RAY_BACKEND_PORT}/generate",
                data=request.json(),
            ) as resp:
                result = await resp.read()
                self.logger.info(
                    f"Response for {request.job_id} (seed={request.seed}): {result}"
                )
                return json.loads(result)


class Scheduler:
    def __init__(
        self,
        pq: PriorityQueue,
        directives: dict,
        logdir: str,
        timer_epoch: float = 0.1,
        max_outstanding_requests: int = 2,
    ):
        self.pq = pq
        self.directives = directives
        self.timer_epoch = timer_epoch
        self.max_outstanding_requests = max_outstanding_requests
        self.waiting_requests = []
        self.waiting_job_map = {}

        self.submitter = AsyncSubmitter.remote(logdir)
        self.response_handler_thread = Thread(target=self.handle_responses)

        self.logger = create_file_logger(
            name=__name__,
            filename=os.path.join(logdir, "scheduler.log"),
            level=logging.DEBUG,
        )

        self.lock = threading.Lock()

    def start(self):
        self.logger.info("Scheduler thread starting!")
        self.response_handler_thread.start()
        self.drain_queue()

    def schedule_job(self, job: Job):
        self.logger.info(f"(Scheduler) scheduling job: {job.job_id}!")
        self.priority_queue.put((job.priority, job))
        self.logger.info(f"(Scheduler) done scheduling job: {job.job_id}")

    def queue_size(self):
        return self.pq.qsize()

    def handle_responses(self):
        while True:
            # Give other thread time
            sleep(self.timer_epoch)
            self.lock.acquire()

            try:
                self.logger.debug(
                    f"(Scheduler:handle_responses:before)priority_queue: {self.pq.get_attribute('queue')}, {self.pq.qsize()}"
                )

                # Check what responses are available
                start = time.time()
                ready, self.waiting_requests = ray.wait(
                    self.waiting_requests,
                    num_returns=len(self.waiting_requests),
                    timeout=0.2,
                    fetch_local=False,
                )
                self.logger.info(
                    f"(Scheduler:handle_response) ray.wait() time total: {time.time() - start}"
                )
                self.logger.debug(
                    f"********* ready={ready}, waiting_requests={self.waiting_requests}"
                )
                ready_job_ids = [response["job_id"] for response in ray.get(ready)]
                self.logger.debug(f"ready_job_ids: {ready_job_ids}")
                # self.logger.debug(f"waiting_job_map: {self.waiting_job_map.keys()}")

                # TODO: fix if/else hack - we need this because on job prioritization, since the current implementation
                # launches a new job, we can have 2 jobs with the same job_id in flight. This is a problem because
                # if they're both in flight at the same time, if the prioritized job finishes first, then it will remove
                # the job_id from the waiting map, so when the other job finishes then it will no longer be in the waiting map...
                ready_jobs = [
                    self.waiting_job_map[job_id]
                    if job_id in self.waiting_job_map
                    else None
                    for job_id in ready_job_ids
                ]

                # Add all jobs with responses back to ready queue
                for job_id, job in zip(ready_job_ids, ready_jobs):
                    # TODO fix if/else hack (see above)
                    if job is None:
                        continue

                    self.logger.info(f"(Scheduler) processing response for {job}")
                    if len(job.subjobs) > 0:
                        self.pq.put((job.priority, job), block=True, timeout=2)
                    else:
                        # TODO: sometimes waiting_job_map does not contain job_id... presumably because it gets
                        # updated asynchronously somehow... need to figure out why/when this happens
                        try:
                            self.logger.debug(f"Removing {job_id} from waiting_job_map")
                            del self.waiting_job_map[job_id]
                        except KeyError:
                            continue
                        except Exception:
                            self.logger.debug(f"wtf: {traceback.format_exc()}")

                self.logger.debug(
                    f"(Scheduler:handle_responses:after) num_waiting_requests: {len(self.waiting_requests)},  priority_queue: {self.pq.get_attribute('queue')}, {self.pq.qsize()}"
                )

            except Exception as e:
                self.logger.debug(f"wtf: {traceback.format_exc()}")

            self.lock.release()

    def drain_queue(self):
        while True:
            # Give other thread time
            sleep(self.timer_epoch)
            self.lock.acquire()

            while len(self.waiting_requests) >= self.max_outstanding_requests:
                self.logger.info(
                    f"(Scheduler:drain_queue) blocking, reached maximum number of outstanding requests ({self.max_outstanding_requests})"
                )
                self.lock.release()
                sleep(self.timer_epoch)
                self.lock.acquire()

            # Get a new job
            self.logger.debug(
                f"(Scheduler:drain_queue:before) priority_queue: {self.pq.get_attribute('queue')}, {self.pq.qsize()}"
            )
            try:
                _, job = self.pq.get(block=True, timeout=2)

                if job.job_id in self.directives:
                    if self.directives[job.job_id] == "finish":
                        self.logger.debug(f"(Scheduler) finishing {job} immediately...")
                        # Since we submit a new job that diffuses the whole image depth first
                        # we need to get rid of the old job that diffuses breadth first
                        # if job.subjobs[0].end_step != job.num_inference_steps:
                        if job.priority != -1:
                            del self.directives[job.job_id]
                            self.lock.release()
                            continue
                    elif self.directives[job.job_id] == "cancel":
                        self.logger.debug(
                            f"(Scheduler) cancelling {job} immediately..."
                        )
                        del self.directives[job.job_id]
                        del self.waiting_job_map[job.job_id]
                        self.lock.release()
                        continue
                    else:
                        raise NotImplementedError

                subjob = job.pop_subjob()
                self.logger.info(
                    f"(Scheduler) got new job/subjob from queue: job: {job}, subjob: {subjob}"
                )
                self.logger.debug(
                    f"(Scheduler:drain_queue:after) priority_queue: {self.pq.get_attribute('queue')}, {self.pq.qsize()}"
                )
            except Empty:
                self.logger.debug("Queue is empty... waiting...")
                self.lock.release()
                continue
            except Exception:
                self.logger.debug(
                    f"(Scheduler:drain_queue) wtf: {traceback.format_exc()}"
                )
                self.lock.release()
                continue

            # Create a request for backend
            if job.name == "similar":
                backend_request = GenerateRequestBackend(
                    job_id=job.job_id,
                    start_step=subjob.start_step,
                    end_step=subjob.end_step,
                    total_steps=job.num_inference_steps,
                    prompt=subjob.diffusion_params.prompt,
                    guidance_scale=subjob.diffusion_params.guidance_scale,
                    seed=subjob.diffusion_params.seed,
                    viz_params=job.viz_params,
                    name=job.name,
                    original_job_id=job.original_job_id,
                )
            elif job.name == "collage":
                print("Converting collage to backend request...")
                backend_request = GenerateRequestBackend(
                    job_id=job.job_id,
                    start_step=subjob.start_step,
                    end_step=subjob.end_step,
                    total_steps=job.num_inference_steps,
                    prompt=subjob.diffusion_params.prompt,
                    guidance_scale=subjob.diffusion_params.guidance_scale,
                    seed=subjob.diffusion_params.seed,
                    viz_params=job.viz_params,
                    name=job.name,
                    collage_layers=job.collage_layers,
                )
            elif job.name == "collage-edit":
                backend_request = GenerateRequestBackend(
                    job_id=job.job_id,
                    start_step=subjob.start_step,
                    end_step=subjob.end_step,
                    total_steps=job.num_inference_steps,
                    prompt=subjob.diffusion_params.prompt,
                    guidance_scale=subjob.diffusion_params.guidance_scale,
                    seed=subjob.diffusion_params.seed,
                    viz_params=job.viz_params,
                    name=job.name,
                    collage_layers=job.collage_layers,
                    collage_src=job.collage_src,
                )
            else:
                backend_request = GenerateRequestBackend(
                    job_id=subjob.job_id,
                    start_step=subjob.start_step,
                    end_step=subjob.end_step,
                    total_steps=job.num_inference_steps,
                    prompt=subjob.diffusion_params.prompt,
                    guidance_scale=subjob.diffusion_params.guidance_scale,
                    seed=subjob.diffusion_params.seed,
                    viz_params=job.viz_params,
                    name=job.name,
                )

            # Submit request and update data structures to track waiting jobs
            self.logger.info(f"(Scheduler) Posting request for {job}...")
            start = time.time()
            self.waiting_requests.append(
                self.submitter.post_async.remote(backend_request)
            )
            self.waiting_job_map[job.job_id] = job
            self.logger.info(
                f"(Scheduler:drain_queue) post_async time total: {time.time() - start}"
            )

            self.pq.task_done()
            self.logger.info("(Scheduler) signaled task is done.")

            self.lock.release()


def run_scheduler(pq: PriorityQueue, directives: dict, timer_epoch: float, logdir: str):
    scheduler = Scheduler(pq, directives, logdir, timer_epoch)
    scheduler.start()


# For testing
if __name__ == "__main__":
    ray.init()
    pq = PriorityQueue()
    run_scheduler(pq, 0.2)
    params = DiffusionParams("hello world", 7, 10)
    j = Job(10, False, params, 2)
    pq.put(j)
