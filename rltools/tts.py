
import time

import ChatTTS
import torch
import torchaudio
import pyaudio
import numpy as np
from scipy.signal import resample


IS_SAVE = False
IS_DISPLAY = True
SEED = 17
CHANNELS = 1
RATE = 16000
FORMAT = pyaudio.paInt16
CHUNK = 4096

resample_factor = RATE / 24000  # chattts采样率是24000，重采样为RATE

chat = ChatTTS.Chat()
chat.load(compile=False)  # Set to True for better performance
torch.manual_seed(SEED)
rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.3,
    top_P=0.7,
    top_K=20,
)

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt="[oral_2][laugh_0][break_6]",
)


texts = ["Put your first text here", "喂,你在吗？"]

wavs = chat.infer(
    texts,
    # skip_refine_text=True,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

print(wavs[0].shape)

if IS_DISPLAY:
    p = pyaudio.PyAudio()
    # 打开麦克风流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK,
    )

    for i in range(len(wavs)):
        signal = wavs[i].squeeze()
        resampled_signal = resample(signal, int(resample_factor * signal.shape[0]))  # 重采样
        output = (resampled_signal * 32768).astype(np.int16)[:, np.newaxis]
        stream.write(output.tobytes())  # np.ndarray to bytes，否则声音卡顿

    stream.stop_stream()
    stream.close()

    p.terminate()

if IS_SAVE:
    for i in range(len(wavs)):
        """
        In some versions of torchaudio, the first line works but in other versions, so does the second line.
        """
        try:
            torchaudio.save(
                f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), RATE
            )
        except:
            torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), RATE)