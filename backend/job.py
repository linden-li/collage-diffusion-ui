from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib

from data_models import Layer


def compute_hash(x: str):
    return hashlib.sha256(bytes(x, "utf-8")).hexdigest()


@dataclass
class DiffusionParams:
    prompt: str
    guidance_scale: float
    seed: int

    def compute_hash(self):
        return compute_hash(f"{self.prompt},{self.guidance_scale},{self.seed}")


@dataclass
class SubJob:
    start_step: int
    end_step: int
    job_id: str
    diffusion_params: DiffusionParams


@dataclass(order=True)
class Job:
    def __init__(
        self,
        num_inference_steps: int,
        breadth_first: bool,
        diffusion_params: DiffusionParams,
        viz_params: bool,
        batch_size: int = 1,
        original_job_id: str = "",
        name: str = "",
        collage_layers: Optional[List[Layer]] = None,
        collage_src: Optional[str] = None,
    ):
        self.priority = 0
        self.subjobs = []
        self.waiting = None  # set to job_id when waiting for a job
        # Stringify the arguments to create a unique job id
        args = f"{num_inference_steps},{breadth_first},{diffusion_params},{viz_params},{batch_size},{original_job_id},{name},{collage_layers}"
        self.job_id = compute_hash(args)
        # self.job_id = diffusion_params.compute_hash()
        self.job_id = f"{name}-{self.job_id}"

        self.diffusion_params = diffusion_params
        self.num_inference_steps = num_inference_steps
        self.viz_params = viz_params
        self.batch_size = batch_size
        self.breadth_first = breadth_first
        self.name = name
        self.original_job_id = original_job_id
        self.collage_layers = collage_layers
        self.collage_src = collage_src

        if breadth_first:
            self.subjobs = [
                SubJob(
                    start_step=i,
                    end_step=min(i + batch_size - 1, num_inference_steps),
                    job_id=self.job_id,  # TODO set this to end_step??
                    diffusion_params=self.diffusion_params,
                )
                for i in range(
                    0, num_inference_steps + batch_size, batch_size
                )  # TODO fix if not multiple of batch size
            ]
            for subjob in self.subjobs:
                print(f"{subjob}")
        else:
            # Create a single sub job if depth-first generation
            self.subjobs = [
                SubJob(
                    start_step=0,
                    end_step=num_inference_steps,
                    job_id=self.job_id,
                    diffusion_params=self.diffusion_params,
                )
            ]

    def __repr__(self):
        return f"(job_id={self.job_id}, seed={self.diffusion_params.seed}, priority={self.priority})"

    def pop_subjob(self) -> SubJob:
        if len(self.subjobs) > 0:
            subjob = self.subjobs.pop(0)
        else:
            return None

        # Update priority to earliest start step
        if len(self.subjobs) > 0:
            self.priority = self.subjobs[0].start_step

        self.waiting = subjob.job_id  # wait on completion of subjob

        return subjob

    def compute_subjob_id(self, end_step: int):
        return self.diffusion_params.compute_hash() + f"_{end_step}"

    # def finish_immediately(self):
    #     self.priority = -1  # TODO figure out this
    #     self.subjobs = [
    #         SubJob(
    #             start_step=self.subjobs[0].start_step,
    #             end_step=self.subjobs[-1].end_step,
    #             job_id=self.compute_subjob_id(self.num_inference_steps),
    #             diffusion_params=self.diffusion_params,
    #         )
    #     ]

    # def cancel(self):
    #     self.subjobs = []
