#!/bin/bash

# Container name
CONTAINER_NAME="reps-mongo-container"

if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  source .env
else
  echo "The .env file does not exist."
fi

# Build and start the container
docker-compose up --build -d

if [ $? -ne 0 ]; then
    echo "There was an error starting the container."
    exit 1
fi

# Check if the container is running
if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep "$CONTAINER_NAME" > /dev/null; then
    echo "The container $CONTAINER_NAME is already running."
fi