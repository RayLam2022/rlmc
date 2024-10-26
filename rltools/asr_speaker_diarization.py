# diarizarion.py

import os

from dotenv import load_dotenv
from pyannote.audio import Model, Pipeline, Inference
from pyannote.core import Segment, Annotation, Timeline
from scipy.spatial.distance import cosine
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

load_dotenv("../app_test/hf.env")
HUGGINGFACE_ACCESS_TOKEN = os.environ["HUGGINGFACE_ACCESS_TOKEN"]

model_id = "openai/whisper-large-v3-turbo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PUNC_SENT_END = ['.', '?', '!']

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# stt_model = WhisperModel(model_size, device=device, compute_type="float16")
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
stt_model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)

generate_kwargs={"language": "chinese",    # 语种 language
                 "logprob_threshold": -1.0,
                 "no_speech_threshold": 0.6,
                 }

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['chunks']:
        start = item['timestamp'][0]
        end = item['timestamp'][1]
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text

def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text

def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


class Diarization:
    def __init__(self):
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

        inference = self.pipeline(
            {
                "waveform": audio_tensor,
                "sample_rate": sample_rate,
                "min_speakers": 1,
                "max_speakers": 5,
            }
        )
        # convert output to list of dicts
        diarization = self.transform_diarization_output(inference)
        return inference, diarization


if __name__ == "__main__":
    target_sample_rate = 16000
    file_path=os.environ["file_path"]

    waveform, sample_rate = torchaudio.load(
        file_path
    )

    resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate) # 重采样
    waveform = resampler(torch.mean(waveform, dim=0, keepdim=True))   # torch.mean(waveform, dim=0, keepdim=True)用于torchaudio读的可能是多通道，转单通道

    result = pipe(waveform[0].numpy(), generate_kwargs=generate_kwargs,return_timestamps=True)
    print(result)
    # embed_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_ACCESS_TOKEN)
    diarization = Diarization()
    inference, diarization_result = diarization.process(waveform, target_sample_rate)
    print(diarization_result)

    res = diarize_text(result, inference)
    print(res)
