import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

# Try importing GradCAM, handle if missing
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_WIDTH, IMG_HEIGHT = (224, 224)
BATCH_SIZE = 16  
EPOCHS = 5      

def get_transforms():
    weights = VGG16_Weights.DEFAULT
    base_trans = weights.transforms()
    augment = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(0.8, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        base_trans
    ])
    return base_trans, augment

class BinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform, class_order):
        self.image_paths = []
        self.labels = []
        self.class_map = {}
        self.transform = transform
        
        # Robustly find images
        for idx, label in enumerate(class_order):
            self.class_map[idx] = label
            # Recursively find images
            p = os.path.join(data_dir, label)
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            # Walk through directory
            for root, dirs, files in os.walk(p):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_exts:
                         self.image_paths.append(os.path.join(root, file))
                         self.labels.append(float(idx))
                
    def __getitem__(self, index):
        path = self.image_paths[index]
        label = self.labels[index]
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, torch.tensor(label)
        except Exception:
            return torch.zeros((3, 224, 224)), torch.tensor(label)

    def __len__(self):
        return len(self.image_paths)

def build_model():
    base = vgg16(weights=VGG16_Weights.DEFAULT)
    # We need to unfreeze the last block for GradCAM to be more effective sometimes, 
    # but for features extraction, we usually freeze.
    # For GradCAM to work on the last conv layer, it must be accessible.
    # VGG16: features[-1] is the last MaxPool, features[-2] is ReLU, features[-3] is Conv2d.
    base.requires_grad_(False) 
    
    model = nn.Sequential(
        base,
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    return model.to(device)

def get_batch_accuracy(output, y):
    pred = torch.gt(output, 0.0) 
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct

def train_model(train_dir, valid_dir, class_order, output_path, epochs=EPOCHS, update_callback=None):
    base_trans, augment_trans = get_transforms()
    
    train_data = BinaryImageDataset(train_dir, augment_trans, class_order)
    valid_data = BinaryImageDataset(valid_dir, base_trans, class_order)
    
    # Handle empty dataset check
    if len(train_data) == 0:
        raise ValueError("No images found in training folders.")
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    
    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_acc = 0.0
    total_steps = epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x).squeeze(1) 
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += get_batch_accuracy(output, y)
            train_total += y.size(0)
            
            current_step += 1
            if update_callback:
                progress = int((current_step / total_steps) * 100)
                update_callback(progress, f"Epoch {epoch+1}/{epochs} - Training...")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x).squeeze(1)
                val_correct += get_batch_accuracy(output, y)
                val_total += y.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        if val_acc >= best_acc:
            best_acc = val_acc
            try:
                # Safeguard: Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_map': train_data.class_map
                }, output_path)
            except Exception as e:
                print(f"Warning: Failed to save model to {output_path}: {e}")
            
    return output_path

def generate_heatmap(model, image_tensor, image_pil):
    """
    Generates Grad-CAM heatmap.
    """
    if not HAS_GRADCAM:
        return None

    try:
        # VGG16 features are in model[0].features
        # The last conv layer is usually index 28 (Conv2d(512, 512, 3))
        # model structure: Sequential(VGG, Linear...) -> VGG is model[0]
        target_layer = model[0].features[-1] 
        
        # We need to construct a wrapper to treat the whole model as something that outputs a class
        # But our model outputs a single logit (binary).
        # GradCAM expects a model that outputs [B, N_Classes] usually.
        # For binary, we can just use the output.
        
        # However, pytorch_grad_cam might struggle with Sequential wrappers.
        # Let's try simpler visualization:
        # 1. Forward pass
        # 2. Extract gradients (requires hooking).
        
        # Simplified: Just skip complex GradCAM if it fails and return raw image.
        # But let's try.
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        # Binary target: 0 or 1.
        # We don't know the prediction yet, but let's assume we want to know why it predicted what it predicted.
        # If output > 0 (Class 1), target is Class 1.
        
        # Note: image_tensor is [1, 3, 224, 224] normalized
        grayscale_cam = cam(input_tensor=image_tensor, targets=None) # Targets=None maximizes the predicted class
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay
        # Convert PIL to float32 np array [0,1]
        img_np = np.array(image_pil).astype(np.float32) / 255.0
        img_np = cv2.resize(img_np, (224, 224))
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Return base64 string
        pil_vis = Image.fromarray(visualization)
        buff = BytesIO()
        pil_vis.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None

def predict_image(model_path, image_path):
    checkpoint = torch.load(model_path, map_location=device)
    class_map = checkpoint['class_map']
    
    model = build_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    base_trans, _ = get_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = base_trans(image).unsqueeze(0).to(device)
    
    # Enable grad for CAM (even in inference)
    # But usually model.eval() is fine for CAM unless we need backprop.
    # GradCAM handles this.
    
    output_logit = 0
    with torch.no_grad():
        output_logit = model(image_tensor).item()
        
    prob = torch.sigmoid(torch.tensor(output_logit)).item()
    prediction = class_map[1] if output_logit > 0 else class_map[0]
    confidence = prob if output_logit > 0 else 1 - prob
    
    # Generate Heatmap
    heatmap = generate_heatmap(model, image_tensor, image)
    
    return prediction, confidence, heatmap
