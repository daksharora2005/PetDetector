# 🐶 PetDetector - Personalized Pet Door Classifier
## Streamlit application: https://petdetector-prototype-daksharora.streamlit.app/

A personalized pet identification system that distinguishes between your pet and other animals using deep learning. This project uses a fine-tuned VGG16 convolutional neural network with a simple Streamlit web app interface for image classification. No data is stored after training, ensuring user privacy.

## 🚀 Features
- ✅ Binary image classification (Your Pet vs. Not Your Pet)
- ✅ Custom training using your own images
- ✅ Live predictions on uploaded test images
- ✅ Image augmentation for better generalization
- ✅ Privacy-focused — no data storage beyond current session
- ✅ Streamlit-based user-friendly interface
- ✅ Software-only prototype, scalable for hardware deployment (e.g., Smart Pet Door)

## 🏗️ Project Structure
PetDetector/
├── app.py # Streamlit App
├── model_utils.py # Model building, training, prediction utilities
├── uploaded_train_data/ # Temporary training/validation data
├── uploaded_test_images/ # Temporary test images
├── Data/ # ImageNet class index (for fun classifier)
├── requirements.txt # Dependencies
└── README.md # This file


## 📦 Installation
1. **Clone the repository:**
git clone https://github.com/your-username/PetDetector.git
cd PetDetector

2. **Install dependencies:**

3. **Run the app:**


## 🏗️ How to Use
### 🔧 Training
- Go to **"Train a Model"** tab.
- Upload images for:
  - **Class 1:** Your pet (e.g., `your_pet`)
  - **Class 2:** Others (e.g., `not_your_pet`)
- Upload both **training** and **validation** images for each class.
- Click **"🚀 Train Now"** to train a personalized model.

### 🔍 Testing
- Switch to **"Test Image"** tab.
- Upload any image to check whether it's your pet or not.

### 🎉 Fun Classifier
- Test with the ImageNet-pretrained VGG16 on any object (e.g., dogs, cats, bears).

## 🧠 Model Details
- Architecture: **VGG16 pretrained on ImageNet**
- Loss Function: **Binary Cross Entropy with Logits**
- Optimizer: **Adam**
- Image Size: **224x224 pixels**
- Includes random rotations, crops, flips, and color jitter for augmentation.

## 🔐 Privacy
- ✅ Uploaded training, validation, and test images are automatically cleared every time the app restarts or retrains.
- ✅ No data is stored permanently.

## ✨ Future Scope
- Hardware integration (e.g., automate pet door with IoT).
- Extension to multi-pet identification.
- Deployment on cloud or edge devices.

## 🤝 Acknowledgements
- Built as part of an internship project at **CodeFirst Technology** under the **AICTE Virtual Internship Program**.
- Uses open-source libraries: **PyTorch**, **Torchvision**, **Streamlit**, **Pillow**, etc.

## 🐾 License
This project is open source and available under the MIT License.
