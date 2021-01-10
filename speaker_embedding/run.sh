torch-model-archiver --model-name speaker_embedding --serialized-file /home/model-server/services/speaker_embedding/resources/baseline_v2_ap.model --handler handler.py --version 1.0.4
mv speaker_embedding.mar ~/model-store
curl -X DELETE "http://localhost:8081/models/speaker_embedding"
curl -X POST "http://localhost:8081/models?model_name=speaker_embedding&url=speaker_embedding.mar&synchronous=true&initial_workers=1&batch_size=16&max_batch_delay=100"
