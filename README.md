# speaker embedding service

## How to run service:

Build docker image: `./create_docker.sh`

Run and create container: `./run_docker.sh`


## How to use service:

1- Curl client:
```bash
curl 127.0.0.1:8080/predictions/speaker_embedding -T resources/shah.wav
```

2- Python client:
```python
import base64
import requests

import numpy as np

with open('resources/shah.wav', 'rb') as f:
    data = f.read()

response = requests.put('http://127.0.0.1:8080/predictions/speaker_embedding', data=data)
speaker_embedding = np.frombuffer(base64.b64decode(response.text), dtype=np.float32)
```

You could send multiple requests concurrently:
```python
import grequests
reqs = (grequests.put('http://127.0.0.1:8080/predictions/speaker_embedding', data=data) for _ in range(1000))
reps = grequests.map(reqs)
```
