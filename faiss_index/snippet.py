import base64
import json

import faiss
import numpy as np
from tqdm import tqdm

with open('representation.json') as f:
    data = json.load(f)

ids = [i[0].split('/')[1] for i in data]
data = np.array(
    [
        np.frombuffer(base64.b64decode(i[1]), dtype=np.float32)
        for i in tqdm(data)
    ]
)

# embedding dimension
d = 512
# for example the number of speakers
n_list = 1125

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)

index.train(data)
index.add(data)

# get top 50 speakers
src_speaker_index = 100
index.search(data[src_speaker_index].reshape(-1, 512), 50)

# write and read index
faiss.write_index(index, 'index.bin')
index = faiss.read_index('index.bin')
