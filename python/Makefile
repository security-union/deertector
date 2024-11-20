# Makefile for building and running the PyTorch detector Docker container

# Variables
IMAGE_NAME = pytorch-detector
CONTAINER_NAME = pytorch-detector-container
TAG = latest
IMAGE = $(IMAGE_NAME):$(TAG)
MODEL_FILE = model.pt
SCRIPT_FILE = detector.py
REQUIREMENTS_FILE = requirements.txt
IMAGE_DIR = app
INPUT_IMAGE = $(IMAGE_DIR)/dogo.jpeg

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE) .

# Run the Docker container
.PHONY: run
run:
	@echo "Running Docker container..."
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/$(IMAGE_DIR):/app/$(IMAGE_DIR) \
		$(IMAGE) /app/$(INPUT_IMAGE)

# Save the Docker image to a tar file (for transferring to Peplink)
.PHONY: save
save:
	@echo "Saving Docker image to tar file..."
	docker save $(IMAGE) > $(IMAGE_NAME).tar

# Load the Docker image from a tar file (on Peplink)
.PHONY: load
load:
	@echo "Loading Docker image from tar file..."
	docker load < $(IMAGE_NAME).tar

# Clean up Docker images and containers
.PHONY: clean
clean:
	@echo "Cleaning up Docker images..."
	docker rmi $(IMAGE) || true

# Push Docker image to Docker Hub (optional)
.PHONY: push
push:
	@echo "Pushing Docker image to Docker Hub..."
	docker push $(IMAGE)
