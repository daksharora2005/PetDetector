from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import asyncio
from typing import List
import json
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

from model import train_model, predict_image, get_transforms
from schemas import TrainRequest, TrainingStatus, PredictionResponse, FunPredictionResponse, ChatMessage, ChatResponse
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global State for Training ===
# In a production multi-user app, use a database.
training_state = {
    "is_training": False,
    "progress": 0,
    "message": "Idle",
    "error": None
}

# === Directories ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
IMAGENET_INDEX_PATH = os.path.join(BASE_DIR, "..", "Data", "imagenet_class_index.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(TRAINED_MODELS_DIR, "pet_model.pth")

# === Helper Functions ===
def update_progress(progress, message):
    training_state["progress"] = progress
    training_state["message"] = message

def run_training_task(class1: str, class2: str):
    global training_state
    training_state["is_training"] = True
    training_state["progress"] = 0
    training_state["error"] = None
    
    try:
        # === Auto-Negative Logic ===
        train_root = os.path.join(DATA_DIR, "train")
        valid_root = os.path.join(DATA_DIR, "valid")
        class_order = [class1, class2]
        
        # Check Class 2 (Negative)
        c2_train_dir = os.path.join(train_root, class2)
        c2_valid_dir = os.path.join(valid_root, class2)
        
        # If Class 2 dir is missing or empty, populate with defaults
        if not os.path.exists(c2_train_dir) or not os.listdir(c2_train_dir):
            update_progress(5, "Auto-populating background images...")
            defaults_src = os.path.join(DATA_DIR, "defaults", "background")
            
            if not os.path.exists(defaults_src) or not os.listdir(defaults_src):
                 # Fallback if setup_defaults didn't run
                 pass 
            else:
                 os.makedirs(c2_train_dir, exist_ok=True)
                 os.makedirs(c2_valid_dir, exist_ok=True)
                 
                 # Copy defaults
                 files = os.listdir(defaults_src)
                 # 80/20 Split
                 split_idx = int(len(files) * 0.8)
                 train_files = files[:split_idx]
                 valid_files = files[split_idx:]
                 
                 for f in train_files:
                     shutil.copy(os.path.join(defaults_src, f), c2_train_dir)
                 for f in valid_files:
                     shutil.copy(os.path.join(defaults_src, f), c2_valid_dir)

        # Verify data exists
        if not os.path.exists(os.path.join(train_root, class1)) or \
           not os.listdir(os.path.join(train_root, class1)):
             raise ValueError(f"Training data missing for {class1}. Please upload images.")
        
        if not os.listdir(c2_train_dir):
             raise ValueError(f"Training data missing for {class2} and no defaults found.")

        train_model(
            train_dir=train_root,
            valid_dir=valid_root,
            class_order=class_order,
            output_path=MODEL_SAVE_PATH,
            epochs=5,
            update_callback=update_progress
        )
        training_state["message"] = "Training Completed Successfully!"
        training_state["progress"] = 100
    except Exception as e:
        training_state["error"] = str(e)
        training_state["message"] = f"Failed: {str(e)}"
    finally:
        training_state["is_training"] = False

# === Endpoints ===

@app.get("/api")
def read_root():
    return {"message": "Pet Detector Backend API"}

@app.get("/status", response_model=TrainingStatus)
def get_status():
    return training_state

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    split: str = Form(...), # 'train' or 'valid'
    classname: str = Form(...)
):
    """
    Uploads images for a specific class and split (train/valid).
    """
    target_dir = os.path.join(DATA_DIR, split, classname)
    os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    for file in files:
        file_location = os.path.join(target_dir, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        count += 1
        
    return {"message": f"Uploaded {count} files to {split}/{classname}"}

@app.post("/train")
async def start_training(background_tasks: BackgroundTasks, request: TrainRequest):
    if training_state["is_training"]:
        return {"message": "Training already in progress."}
        
    background_tasks.add_task(run_training_task, request.class1_name, request.class2_name)
    return {"message": "Training started in background."}

# === Fun Classifier ===
@app.post("/predict-fun", response_model=FunPredictionResponse)
async def predict_fun(file: UploadFile = File(...)):
    # Save temp file
    temp_path = os.path.join(DATA_DIR, "temp_fun.jpg")
    with open(temp_path, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load MobileNetV2 (Lightweight, ~14MB vs 500MB VGG16)
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    
    # Load Imagenet classes
    # Check if we can find the json
    if os.path.exists(IMAGENET_INDEX_PATH):
        with open(IMAGENET_INDEX_PATH, "r") as f:
            class_idx = json.load(f)
            # Flatten to just labels
            # dict is "0": ["n01440764", "tench"]
            labels = {int(k): v[1] for k,v in class_idx.items()}
    else:
        # Fallback if file missing
        labels = weights.meta["categories"]

    preprocess = weights.transforms()
    img = Image.open(temp_path).convert("RGB")
    batch = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
    
    top5_prob, top5_catid = torch.topk(prediction, 5)
    
    results = []
    for i in range(5):
        label = labels[top5_catid[i].item()]
        prob = top5_prob[i].item()
        results.append({"label": label, "probability": prob})
        
    return {"predictions": results}

# === Chatbot Endpoint ===
from chatbot import bot, PetGuardBot

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(msg: ChatMessage):
    response = bot.get_response(msg.message)
    return {"response": response}

# === Webcam Endpoint ===
import base64
from io import BytesIO

class WebcamRequest(BaseModel):
    image_base64: str

@app.post("/predict-webcam", response_model=PredictionResponse)
async def predict_webcam(request: WebcamRequest):
    if not os.path.exists(MODEL_SAVE_PATH):
         return {"class_name": "Error", "confidence": 0.0, "error": "Model not trained yet."}
         
    # Decode base64
    try:
        header, encoded = request.image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        
        temp_path = os.path.join(DATA_DIR, "temp_webcam.jpg")
        with open(temp_path, "wb") as f:
            f.write(image_data)
            
        pred, conf, heatmap = predict_image(MODEL_SAVE_PATH, temp_path)
        return {"class_name": pred, "confidence": conf, "heatmap_base64": heatmap}
    except Exception as e:
        return {"class_name": "Error", "confidence": 0.0, "error": str(e)}

# === Update Predict for Heatmap ===
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_SAVE_PATH):
        return {"class_name": "Error", "confidence": 0.0, "error": "Model not trained yet."}
    
    # Save temp file
    temp_path = os.path.join(DATA_DIR, "temp_predict.jpg")
    with open(temp_path, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    pred, conf, heatmap = predict_image(MODEL_SAVE_PATH, temp_path)
    return {"class_name": pred, "confidence": conf, "heatmap_base64": heatmap}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
