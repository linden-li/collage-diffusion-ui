from typing import Optional, Dict, List, Union
from pydantic import BaseModel
import re


def to_camel(string: str) -> str:
    return "".join(word.capitalize() for word in string.split("_"))


def to_snake(string: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()


class GenerateRequest(BaseModel):
    prompt: str
    numInferenceSteps: int
    guidanceScale: float
    seed: int
    breadthFirst: bool
    vizParams: bool
    prioritize: bool
    originalJobId: Optional[str]


class Position(BaseModel):
    x: float
    y: float


class Transform(BaseModel):
    position: Position
    scale: float
    rotation: float


class Layer(BaseModel):
    id: int
    key: str
    originalImgUrl: str
    textPrompt: str
    cacStrength: float
    negativeStrength: float
    noiseStrength: float
    cannyStrength: float
    transform: Transform
    polygon: List[Position]
    wordEmbedding: Optional[str]


class CollageRequest(BaseModel):
    layers: List[Layer]
    prompt: str
    numInferenceSteps: int
    guidanceScale: int
    seed: int
    # breadthFirst: bool
    # vizParams: bool
    # prioritize: bool


class CollageEditRequest(BaseModel):
    collageSrc: str  # URL to background of collage
    layers: List[Layer]
    prompt: str
    numInferenceSteps: int
    guidanceScale: int
    seed: int
    breadthFirst: bool
    vizParams: bool
    prioritize: bool


class CollageResponse(BaseModel):
    job_id: str


class FinetuneRequest(BaseModel):
    layers: List[Layer]
    layerIdx: int


class FinetuneResponse(BaseModel):
    embedding_path: str  # path to the embedding file


class DirectiveRequest(BaseModel):
    job_id: str
    directive: str


class DirectiveResponse(BaseModel):
    job_id: str
    directive: str


class GenerateRequestBackend(BaseModel):
    job_id: str
    start_step: int
    end_step: int
    total_steps: int
    prompt: str
    guidance_scale: float
    seed: int
    viz_params: bool
    name: str
    original_job_id: Optional[str]
    collage_layers: Optional[List[Layer]]
    collage_src: Optional[str]


class GenerateResponse(BaseModel):
    job_id: str
    request: GenerateRequest
    data: Optional[Dict]


class PollRequest(BaseModel):
    job_ids: List[str]


class PollResponse(BaseModel):
    job_ids: List[str]
    latest_steps: List[int]


class TokenData(BaseModel):
    data: Union[str, None]
    layerKey: Union[str, None]
    id: str

class CollageState(BaseModel):
    collage_id: str
    collage_prompt: str
    num_inference_steps: int
    guidance_scale: float
    seed: int
    num_frames: int
    layers: List[Layer]
    input_token_data: List[Union[TokenData, None]]
    layer_to_index_map: Dict[str, int]


class CollageStatePushResponse(BaseModel):
    collage_id: str