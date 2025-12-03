from typing import List, Union, Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import handler  # usa el handler.py del contenedor base


class DetectRequest(BaseModel):
    image: Union[str, List[str]]
    purpose: Literal["insightface", "combined_embedding"] = "insightface"


class RunpodStyleRequest(BaseModel):
    input: DetectRequest


app = FastAPI(title="InsightFace API", version="1.0.0")


@app.post("/detect")
async def detect(payload: DetectRequest):
    """
    Endpoint directo:
    POST /detect
    { "image": "<http(s) url> | <base64 string>", "purpose": "insightface" | "combined_embedding" }
    """
    event = {"input": payload.dict()}
    result = handler.handler(event)
    return {"result": result}


@app.post("/runpod")
async def runpod_compatible(payload: RunpodStyleRequest):
    """
    Endpoint compatible con el formato original de RunPod:
    POST /runpod
    { "input": { "image": "<http(s) url> | <base64 string>", "purpose": "..." } }
    """
    event = payload.dict()
    result = handler.handler(event)
    return {"result": result}
