# run speaker embedding service
docker run -d \
    --name speaker_embedding_service \
    -p 8080-8082:8080-8082 \
    speaker-embedding-service:latest
# register models
docker exec speaker_embedding_service /bin/bash "/home/model-server/services/speaker_embedding/run.sh"
