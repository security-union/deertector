# Makefile for building and running the PyTorch detector Docker container

# Variables
GIT_HASH := $(shell git rev-parse --short HEAD)
IMAGE_NAME = securityunion/deertector
CONTAINER_NAME = pytorch-detector-container
DEPENDENCIES_IMAGE = securityunion/deertector-dependencies
TAG = latest
TAG_WITH_HASH = $(TAG)-$(GIT_HASH)
IMAGE = $(IMAGE_NAME):$(TAG_WITH_HASH)
MODEL_FILE = model.pt
SCRIPT_FILE = detector.py
REQUIREMENTS_FILE = requirements.txt
INPUT_IMAGE = /app/images/deer.jpg /app/images/neg-2.jpeg /app/images/neg-3.jpg /app/images/neg.jpeg
LOCAL_IMAGE_DIR = images

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker buildx build --pull=false --platform linux/amd64 -t $(IMAGE) . --load
	docker tag $(IMAGE) $(IMAGE_NAME):$(TAG)

# Run the Docker container locally with volume binding
.PHONY: run-local
run-local:
	@echo "Running Docker container locally..."
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/$(LOCAL_IMAGE_DIR):/app/images \
		$(IMAGE) python /app/detector.py $(INPUT_IMAGE)

# Run the Docker container
.PHONY: run
run:
	@echo "Running Docker container..."
	docker run --rm \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(TAG) /app/detector.py $(INPUT_IMAGE)

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
	docker rmi $(IMAGE_NAME):$(TAG) || true

# Push Docker image to Docker Hub
.PHONY: push
push:
	@echo "Pushing Docker image to Docker Hub..."
	docker push $(IMAGE)
	docker push $(IMAGE_NAME):$(TAG)

# Build the dependencies image
.PHONY: build-dependencies
build-dependencies:
	@echo "Building dependencies image..."
	docker buildx build --platform linux/amd64 -t $(DEPENDENCIES_IMAGE):$(TAG_WITH_HASH) -f Dockerfile.dependencies . --load
	docker tag $(DEPENDENCIES_IMAGE):$(TAG_WITH_HASH) $(DEPENDENCIES_IMAGE):$(TAG)

# Push the dependencies image
.PHONY: push-dependencies
push-dependencies:
	@echo "Pushing dependencies image..."
	docker push $(DEPENDENCIES_IMAGE):$(TAG_WITH_HASH)
	docker push $(DEPENDENCIES_IMAGE):$(TAG)
