#!/bin/bash

# Define the container name
CONTAINER_NAME="reps-frontend-container"

# Check if the container is running
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
  echo "The container $CONTAINER_NAME is running. Stopping it..."
  docker stop $CONTAINER_NAME
else
  echo "The container $CONTAINER_NAME is not running."
fi
