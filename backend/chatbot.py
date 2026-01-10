import os
import random
import requests
import json

# === Mock Logic (Fallback) ===
class MockGuardBot:
    def get_response(self, message: str) -> str:
        msg = message.lower()
        if any(w in msg for w in ["hi", "hello", "hey", "start"]):
            return "Hello! I'm GuardBot (Demo Answer). I can help you with training or webcam setup. (API Unavailable)"
        if "train" in msg or "model" in msg:
            return "To train: Upload 20+ photos of your pet in the 'Train' tab. We auto-generate the background class for you!"
        if "webcam" in msg or "camera" in msg or "live" in msg:
            return "Our Live Guard feature allows real-time pet detection directly in your browser."
        return "I'm currently operating in Offline Mode. Please check the 'Train' tab to get started with the core features."

# === Real Logic (REST API) ===
class PetGuardBot:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDJCJhY_ltyqQA7UuLiDcMJa8y29v9AO4g")
        self.mock = MockGuardBot()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        
        # Models to try in order (Updated based on key availability)
        self.models = [
            "gemini-2.0-flash", 
            "gemini-2.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash" # Keep as backup
        ]

    def get_response(self, message: str) -> str:
        if not self.api_key:
            return self.mock.get_response(message)

        payload = {
            "contents": [{
                "parts": [{"text": f"You are GuardBot, a helpful assistant for PetGuard Pro. Answer this user question concisely: {message}"}]
            }]
        }
        
        # Try models in sequence
        for model in self.models:
            try:
                url = self.base_url.format(model=model, key=self.api_key)
                resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Parse response
                    # Structure: candidates[0].content.parts[0].text
                    if "candidates" in data and data["candidates"]:
                         text = data["candidates"][0]["content"]["parts"][0]["text"]
                         return text
                else:
                    # error
                    pass # Try next model
            except Exception as e:
                print(f"Error calling {model}: {e}")
                continue
        
        # If all fail
        return self.mock.get_response(message)

bot = PetGuardBot()
