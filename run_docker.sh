# run speaker embedding service
docker run -d \
    --name speaker-embedding-service-cpu \
    -p 8080-8082:8080-8082 \
    speaker-embedding-cpu:latest
