from pydantic import BaseModel
from typing import List, Optional

class TrainRequest(BaseModel):
    class1_name: str
    class2_name: str

class TrainingStatus(BaseModel):
    is_training: bool
    progress: int
    message: str
    error: Optional[str] = None

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    heatmap_base64: Optional[str] = None # Added for Explainable AI

class FunPredictionResponse(BaseModel):
    predictions: List[dict] 

class ChatMessage(BaseModel):
    message: str
    history: List[dict] = [] # List of {role: "user"|"assistant", content: str}

class ChatResponse(BaseModel):
    response: str
