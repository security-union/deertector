# Application Dockerfile
FROM securityunion/deertector-dependencies:latest-36c6c86

# Set the working directory
WORKDIR /app

# Copy application code
COPY images /app/images
COPY detector.py /app/detector.py
COPY model.pt /app/model.pt

# Set the entry point
CMD ["python", "detector.py"]
