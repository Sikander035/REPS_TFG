#!/bin/bash

# Define the image name and container name
IMAGE_NAME="reps-backend-image"
CONTAINER_NAME="reps-backend-container"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Check if the container already exists
if [ $(docker ps -a -q -f name=$CONTAINER_NAME) ]; then
  echo "The container $CONTAINER_NAME already exists."
  # Check if the container is stopped
  if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "The container $CONTAINER_NAME is already running."
  else
    echo "The container $CONTAINER_NAME exists but is stopped. Starting it..."
    docker start $CONTAINER_NAME
  fi
else
  echo "The container $CONTAINER_NAME does not exist. Creating and starting it..."
  docker run -d --name $CONTAINER_NAME $IMAGE_NAME
fi