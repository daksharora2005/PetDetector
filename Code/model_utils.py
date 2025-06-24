import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms.functional import to_pil_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_WIDTH, IMG_HEIGHT = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def get_transforms():
    weights = VGG16_Weights.DEFAULT
    base_trans = weights.transforms()
    augment = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(0.8, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    return base_trans, augment


class BinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform, class_order):
        self.imgs = []
        self.labels = []
        self.class_map = {}
        self.transform = transform

        assert len(class_order) == 2, "Expected exactly 2 classes for binary classification."

        for idx, label in enumerate(class_order):
            self.class_map[idx] = label
            image_paths = glob.glob(os.path.join(data_dir, label, "*.jpg"))
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                self.imgs.append(self.transform(img).to(device))
                self.labels.append(torch.tensor(float(idx)).to(device))

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.imgs)


def build_model():
    base = vgg16(weights=VGG16_Weights.DEFAULT)
    base.requires_grad_(False)  # Freeze base
    model = nn.Sequential(
        base,
        nn.Linear(1000, 1)
    )
    return model.to(device)


def get_batch_accuracy(output, y, N):
    pred = torch.gt(output, torch.tensor([0.0]).to(device))
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train_model(model, dataloader, val_loader, augment, class_order, epochs=EPOCHS):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_N = len(dataloader.dataset)
    val_N = len(val_loader.dataset)

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0

        for x, y in dataloader:
            x_aug = augment(x)
            output = model(x_aug).squeeze()
            optimizer.zero_grad()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += get_batch_accuracy(output, y, train_N)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {total_acc:.4f}")
        validate(model, val_loader, loss_fn, val_N)

    # === Fine-tuning ===
    print("\n🔧 Fine-tuning VGG16 base...")
    model[0].requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    for epoch in range(2):
        model.train()
        total_loss, total_acc = 0.0, 0.0

        for x, y in dataloader:
            x_aug = augment(x)
            output = model(x_aug).squeeze()
            optimizer.zero_grad()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += get_batch_accuracy(output, y, train_N)

        print(f"[Finetune] Epoch {epoch+1}/2 | Loss: {total_loss:.4f} | Accuracy: {total_acc:.4f}")
        validate(model, val_loader, loss_fn, val_N)

    return model


def validate(model, val_loader, loss_fn, N):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x).squeeze()
            loss = loss_fn(output, y)
            total_loss += loss.item()
            total_acc += get_batch_accuracy(output, y, N)
    print(f"[Validation] Loss: {total_loss:.4f} | Accuracy: {total_acc:.4f}")


def predict_image(model, image_path, transform, class_map):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image).item()
    return class_map[0] if output < 0 else class_map[1]
