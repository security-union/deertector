# Use a lightweight Python base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Add the detector script
COPY model.pt /app/model.pt
COPY detector.py /app/detector.py
COPY images /app/images

# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT ["python", "detector.py"]
