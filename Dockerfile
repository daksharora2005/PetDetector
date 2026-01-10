# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first to cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend
COPY backend/ backend/
COPY start_app.bat .

# Copy the frontend build (Pre-built)
# Users must run 'build_for_prod.bat' locally first to generate backend/static
# OR we can assume the user will push the static files?
# Actually HF builds from source usually. Let's make it simple:
# We will use the Unified Server strategy.
# But HF doesn't run NPM easily without multi-stage build.
# Simplify: We assume 'backend/static' is populated or we strictly use Python.
# Let's add Node to build frontend in Docker (Best practice).

# Install Node.js
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy Frontend
COPY frontend/ frontend/
WORKDIR /app/frontend
RUN npm install && npm run build
WORKDIR /app

# Move Build
RUN mkdir -p backend/static && cp -r frontend/dist/* backend/static/

# Permissions
RUN chmod -R 777 /app

# Hugging Face Spaces listens on port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Command
CMD ["python", "backend/production.py"]
