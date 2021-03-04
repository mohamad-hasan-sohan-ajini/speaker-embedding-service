# run speaker embedding service
docker run -d \
    --name speaker-embedding-service-gpu \
    --gpus all \
    -p 8080-8082:8080-8082 \
    speaker-embedding-gpu:latest
