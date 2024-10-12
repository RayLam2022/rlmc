# diarizarion.py

import os

from dotenv import load_dotenv
from pyannote.audio import Model, Pipeline, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cosine
import torch
import torchaudio

load_dotenv("../app_test/hf.env")
HUGGINGFACE_ACCESS_TOKEN = os.environ["HUGGINGFACE_ACCESS_TOKEN"]


class Diarization:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_ACCESS_TOKEN
        ).to(device)

    def transform_diarization_output(self, diarization):
        l = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            l.append({"start": segment.start, "end": segment.end, "speaker": speaker})
        return l

    def process(self, waveform, sample_rate):
        audio_tensor = waveform
        print(audio_tensor.shape)

        diarization = self.pipeline(
            {
                "waveform": audio_tensor,
                "sample_rate": sample_rate,
                "min_speakers": 1,
                "max_speakers": 5,
            }
        )
        # convert output to list of dicts
        diarization = self.transform_diarization_output(diarization)
        return diarization


if __name__ == "__main__":
    waveform, sample_rate = torchaudio.load(
        ''
    )
    # embed_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_ACCESS_TOKEN)
    diarization = Diarization()
    res = diarization.process(waveform, sample_rate)
    print(res)
