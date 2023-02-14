import base64
import io
import json
import os
import subprocess
import sys

import ffmpeg
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

sys.path.append('/home/model-server/services/speaker_embedding/voxceleb_trainer')
from SpeakerNet import SpeakerNet

MODEL_PATH = '/home/model-server/services/speaker_embedding/resources/baseline_v2_ap.model'
CONFIG_PATH = '/home/model-server/services/speaker_embedding/resources/config.json'


class SpeakerEmbeddingHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.sample_rate = 16000

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

    def convert_audio_to_wav(self, audio_content):
        args = (
            ffmpeg
            .input('pipe:')
            .output(
                'pipe:',
                format='wav',
                acodec='pcm_s16le',
                ac=1,
                ar=self.sample_rate
            )
            .get_args()
        )
        ffmpeg_process = subprocess.Popen(
            ['ffmpeg'] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        wav_content = ffmpeg_process.communicate(input=audio_content)[0]
        wav_content = io.BytesIO(wav_content)
        ffmpeg_process.kill()
        return wav_content
    
    def load_wav(self, filename, max_frames=400):
        wav_content = self.convert_audio_to_wav(filename)
        audio, _ = sf.read(wav_content)
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
            if self.model.__L__.test_normalize:
                out = F.normalize(out, p=2, dim=1)
        out = [
            F.normalize(out[start:end].mean(dim=0), dim=0).cpu().numpy()
            for start, end in zip(length[:-1], length[1:])
        ]
        return out

    def handle(self, batch, context):
        batch = [self.load_wav(i.get('body')) for i in batch]
        result = self.inference_batch(batch)
        return [base64.b64encode(embedding.tobytes()) for embedding in result]
