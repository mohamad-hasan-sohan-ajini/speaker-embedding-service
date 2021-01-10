# check if model exists
[ ! -d "speaker_embedding/resources" ] && mkdir speaker_embedding/resources
if [ -f "speaker_embedding/resources/baseline_v2_ap.model" ]; then
    echo "speaker embedding model exists."
else
    wget http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model -O speaker_embedding/resources/baseline_v2_ap.model
fi

docker build -t speaker-embedding-service .
