import os
import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add current directory to path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app

# Directory where frontend build will be placed
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Mount Static Files (JS, CSS, Images)
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")
    # Also mount public/demo.mp4 if it was copied to root of dist, usually it is.
    # Vite copies public/* to dist root.
    
    @app.get("/{full_path:path}")
    async def serve_spa_or_static(full_path: str):
        # Check if it matches a file in static dir (e.g. demo.mp4, vite.svg)
        potential_path = os.path.join(STATIC_DIR, full_path)
        if os.path.exists(potential_path) and os.path.isfile(potential_path):
            return FileResponse(potential_path)
            
        # Otherwise serve index.html for SPA routing
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    print(f"✅ Production Server Ready: Serving Frontend from {STATIC_DIR}")
else:
    print("⚠️  Static directory not found. Run 'build_for_prod.bat' first.")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    # Listen on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=port)
