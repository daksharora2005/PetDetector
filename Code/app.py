import streamlit as st
import os
import shutil
from PIL import Image
import torch
import json

from model_utils import (
    get_transforms,
    BinaryImageDataset,
    build_model,
    train_model,
    predict_image
)

# === CONSTANTS ===
TRAIN_DIR = "uploaded_train_data"
TEST_IMG_DIR = "uploaded_test_images"
IMAGENET_JSON = os.path.join("Data", "imagenet_class_index.json")

# === CLEANUP ON LAUNCH ===
for folder in [TRAIN_DIR, TEST_IMG_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(folder, "valid"), exist_ok=True)

# === TRANSFORMS ===
pre_trans, augment_trans = get_transforms()

# === UI HEADER ===
st.set_page_config(page_title="Personalized Pet Door", layout="wide")
st.title("🐶 Personalized Pet Door")
st.markdown("---")

# === SIDEBAR NAV ===
tab = st.sidebar.radio("Choose an action:", ["Train a Model", "Test Image", "Fun Classifier (ImageNet)"])


# ===============================
# TRAINING TAB
# ===============================
if tab == "Train a Model":
    st.header("📦 Upload Training and Validation Data")

    class1_name = st.text_input("Class 1 Name", value="", placeholder="e.g. your pet")
    class2_name = st.text_input("Class 2 Name", value="", placeholder="e.g. not your pet")

    class1_train = st.file_uploader(f"Train images - {class1_name}", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="class1_train") if class1_name else []
    class1_valid = st.file_uploader(f"Validation images - {class1_name}", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="class1_valid") if class1_name else []

    class2_train = st.file_uploader(f"Train images - {class2_name}", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="class2_train") if class2_name else []
    class2_valid = st.file_uploader(f"Validation images - {class2_name}", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="class2_valid") if class2_name else []

    st.markdown("### ✅ Step 3: Train the model")

    if st.button("🚀 Train Now"):
        if not all([class1_name, class2_name, class1_train, class2_train]):
            st.error("Please upload training images and provide both class names before training.")
            st.stop()

        # Save fresh images
        def save_images(file_list, split, class_name):
            dest = os.path.join(TRAIN_DIR, split, class_name)
            os.makedirs(dest, exist_ok=True)
            for file in file_list:
                with open(os.path.join(dest, file.name), "wb") as f:
                    f.write(file.getbuffer())

        save_images(class1_train, "train", class1_name)
        save_images(class1_valid, "valid", class1_name)
        save_images(class2_train, "train", class2_name)
        save_images(class2_valid, "valid", class2_name)

        # Verify structure
        train_root = os.path.join(TRAIN_DIR, "train")
        valid_root = os.path.join(TRAIN_DIR, "valid")
        if len(os.listdir(train_root)) != 2:
            st.warning("You must upload exactly two distinct class folders.")
        else:
            class_order = [class1_name, class2_name]
            train_data = BinaryImageDataset(train_root, pre_trans, class_order)
            valid_data = BinaryImageDataset(valid_root, pre_trans, class_order)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
            val_loader = torch.utils.data.DataLoader(valid_data, batch_size=16)

            model = build_model()
            st.text("Training started...")
            trained_model = train_model(model, train_loader, val_loader, augment_trans, class_order)
            st.session_state["model"] = trained_model
            st.session_state["class_map"] = train_data.class_map
            st.success("🎉 Model trained and ready!")


# ===============================
# TESTING TAB
# ===============================
elif tab == "Test Image":
    st.header("🔍 Upload Test Image")

    if "model" not in st.session_state:
        st.warning("You need to train a model first in the 'Train a Model' tab.")
    elif "class_map" not in st.session_state:
        st.error("Class mapping not found. Please re-train the model.")
    else:
        uploaded_image = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            test_img_path = os.path.join(TEST_IMG_DIR, uploaded_image.name)
            with open(test_img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.image(test_img_path, caption="Uploaded Test Image", use_container_width=True)

            predicted_class = predict_image(
                st.session_state["model"],
                test_img_path,
                pre_trans,
                st.session_state["class_map"]
            )
            st.success(f"Prediction: **{predicted_class}**")


# ===============================
# FUN CLASSIFIER TAB
# ===============================
elif tab == "Fun Classifier (ImageNet)":
    st.header("🎲 Fun VGG16 Classifier (Dogs, Cats, Bears, etc.)")

    with open(IMAGENET_JSON, "r") as f:
        imagenet_classes = json.load(f)

    vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    vgg_model.eval()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="fun_upload")

    if uploaded:
        img_path = os.path.join(TEST_IMG_DIR, "fun.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.image(img_path, caption="Image for Fun Classifier", use_container_width=True)

        image = Image.open(img_path).convert("RGB")
        image = pre_trans(image).unsqueeze(0)

        with torch.no_grad():
            output = vgg_model(image)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top3 = torch.topk(probs, 3)

        st.subheader("Top 3 Predictions")
        for idx in top3.indices:
            cls_id = str(idx.item())
            label = imagenet_classes[cls_id][1]
            prob = probs[idx].item()
            st.write(f"**{label}** — {prob:.2%}")
