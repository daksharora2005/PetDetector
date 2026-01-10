# � PetGuard Pro
**The AI-Powered Sentry for Your Beloved Pets.**

**PetGuard Pro** is a next-generation smart pet door system that uses advanced computer vision to ensure only *your* pet enters your home. It combines a self-learning neural network with a beautiful, real-time dashboard and an intelligent AI assistant.

---

## ✨ Key Features

### 🛡️ Smart Security (Zero-Trust)
*   **Facial Recognition**: Fine-tuned VGG16 model that learns to recognize your specific pet in under 2 minutes.
*   **Auto-Rejection**: Instantly locks the door for raccoons, stray cats, and unknown animals.
*   **Explainable AI**: View **Heatmaps (Grad-CAM)** to see exactly *why* the AI made a decision (e.g., focusing on the snout or ears).

### 🖥️ Modern Dashboard (React + Vite)
*   **Dark Mode Aesthetic**: A professional, "Cosmic Dark" UI with glassmorphism and bento-grid layouts.
*   **Live Surveillance**: Real-time webcam feed with predicted confidence overlays.
*   **Training Panel**: Drag-and-drop interface to teach the AI new faces instantly.

### 🤖 Intelligent Assistant (Gemini AI)
*   **PetGuard Bot**: A built-in chatbot powered by **Google Gemini 2.0 Flash**.
*   **Context Aware**: Ask questions like *"Did you see any raccoons tonight?"* or *"How do I improve accuracy?"*.

---

## 🏗️ Tech Stack

*   **Frontend**: React, Tailwind CSS, Framer Motion, Axios.
*   **Backend**: FastAPI, Uvicorn, Python 3.9+.
*   **Machine Learning**: PyTorch (MobileNetV2 Transfer Learning), Grad-CAM.
*   **Deployment**: Dockerized for Hugging Face Spaces (or any Container service).

---

## ☁️ Deployment (Recommended)

## ☁️ Cloud Deployment (Hugging Face Spaces) - Recommended

We use **Hugging Face Spaces** because it offers generous **16GB RAM** for free, which is perfect for running AI models like PetGuard.

1.  **Create a Space**:
    *   Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    *   Select **Docker** as the SDK.
    *   Choose "Blank" template.
    *   Set visibility to **Public**.

2.  **Deploy Code**:
    *   **Option A**: Connect this GitHub repository in the Space settings.
    *   **Option B**: Drag and drop all project files into the Space's "Files" tab.

3.  **Add Your API Key**:
    *   Go to Space **Settings** -> **Variables and secrets**.
    *   Add a new Secret: `GEMINI_API_KEY` with your Google Gemini key.

The app will build automatically (takes ~5 mins) and launch! 🚀

---

## 🚀 Local Desktop Setup

### Prerequisites
*   Python 3.9+
*   Node.js & npm

### 1. Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Run the App
**Option A: One-Click Production (Best for Demos)**
Double-click `build_for_prod.bat`.
*   This builds the React app and serves it via Python at `http://localhost:8000`.

**Option B: Developer Mode**
*   Backend: `python backend/main.py`
*   Frontend: `npm run dev` (in frontend folder)

---

## � Screenshots

| Landing Page | Live Detection |
|:---:|:---:|
| *Beautiful hero section with 3D elements* | *Real-time inference with heatmaps* |

| Training | Chatbot |
|:---:|:---:|
| *One-click training interface* | *AI Assistant integration* |

---

## 🤝 Contributing
Built with ❤️ as part of the AICTE Virtual Internship.
Open source under the **MIT License**.
