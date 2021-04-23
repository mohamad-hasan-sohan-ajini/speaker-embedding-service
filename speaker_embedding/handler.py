import base64
import io
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from ts.torch_handler.base_handler import BaseHandler

sys.path.append('/home/model-server/services/speaker_embedding/voxceleb_trainer')
from SpeakerNet import SpeakerNet

MODEL_PATH = '/home/model-server/services/speaker_embedding/resources/baseline_v2_ap.model'
CONFIG_PATH = '/home/model-server/services/speaker_embedding/resources/config.json'


class SpeakerEmbeddingHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False

    def initialize(self, context):
        self._context = context
        properties = context.system_properties
        if properties.get("gpu_id") is not None:
            self.device = torch.device("cuda:" + str(properties.get("gpu_id")))
        else:
            self.device = torch.device('cpu')
        with open(CONFIG_PATH) as f:
            kwargs = json.load(f)
        self.model = SpeakerNet(**kwargs)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        self.model.__S__.to(self.device)
        self.model.__L__.to(self.device)
        self.model.eval()
        self.initialized = True

    def load_wav(self, filename, max_frames=400):
        audio, _ = sf.read(filename)
        segment_size = max_frames * 160 + 240
        print(f'segment_size: {segment_size}')
        audio_size = audio.shape[0]
        print(f'audio_size-before: {audio_size}')
        if audio_size <= segment_size:
            print('shortage')
            shortage = segment_size - audio_size
            audio = np.pad(audio, (0, shortage), 'wrap')
            audio_size = audio.shape[0]
        print(f'audio_size-after: {audio_size}')
        num_eval = min(16, (audio_size - segment_size) // 4000 + 1)
        print(f'num_eval: {num_eval}')
        startframe = np.linspace(0, audio_size-segment_size, num_eval)
        print(f'startframe: {startframe}')
        feats = []
        print('chunks')
        for asf in startframe:
            print(f'{int(asf)}:{int(asf)+segment_size}')
            feats.append(audio[int(asf):int(asf)+segment_size])
        feat = np.stack(feats, axis=0).astype(np.float)
        print(f'feat size: {feat.shape}')
        return torch.FloatTensor(feat)

    def inference_batch(self, batch):
        length = np.cumsum([0] + [len(i) for i in batch])
        batch = torch.cat(batch).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            out = [
                out[start:end].mean(dim=0)
                for start, end in zip(length[:-1], length[1:])
            ]
            if self.model.__L__.test_normalize:
                out = [
                    torch.nn.functional.normalize(i, p=2, dim=0).cpu().numpy()
                    for i in out
                ]
        return out

    def handle(self, batch, context):
        batch = [self.load_wav(io.BytesIO(i.get('body'))) for i in batch]
        result = self.inference_batch(batch)
        return [base64.b64encode(embedding.tobytes()) for embedding in result]
