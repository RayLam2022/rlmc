"""
@File    :   assistant_v2.py
@Time    :   2024/07/04 12:21:43
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

from typing import Callable, Optional, Union, List
import multiprocessing as mp
import os
import os.path as osp
import time
import copy
import argparse

import pynput
import cv2
import pyaudio
import pyttsx3
import numpy as np
import torch
from PIL import ImageGrab
# from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import qianfan

parser = argparse.ArgumentParser("assistant")
parser.add_argument(
    "-a",
    "--access_key",
    required=True,
)
parser.add_argument(
    "-s",
    "--secret_key",
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    default="ERNIE-Bot-turbo",  # ERNIE-4.0-8K ChatLaw
    help="ERNIE-4.0-8K, ChatLaw, ERNIE-Bot-turbo...",
)

parser.add_argument(
    "-l",
    "--record_src_language",
    default="chinese", #"zh",
    help="record src language",
)

parser.add_argument(
    "-sp",
    "--is_speak",
    action="store_true",
    help="assistant speak",
)


args = parser.parse_args()

os.environ["QIANFAN_ACCESS_KEY"] = args.access_key
os.environ["QIANFAN_SECRET_KEY"] = args.secret_key


INT16_MAX_ABS_VALUE = 32768.0
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
unit = RATE // CHUNK

# model_id = "large-v3"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# stt_model = WhisperModel(model_id, device=device, compute_type="float16")

model_id = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
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

generate_kwargs={"language": "chinese",    
                 "logprob_threshold": -1.0,
                 "no_speech_threshold": 0.6,
                 "return_timestamps": False,}


def on_press_wrapper(signal: mp.Value) -> Callable:
    def on_press(key: pynput.keyboard.Key) -> Optional[bool]:
        nonlocal signal
        if key == pynput.keyboard.Key.esc:
            signal.value = -1
            return False
        elif hasattr(key, "char"):
            if key.char == "f":
                # pyttsx3.speak('Hello, World!我们的花朵是花园')
                print("start record...")
                signal.value = 1
            if key.char == "s":
                print("end record...")
                signal.value = 0

    return on_press
    # print("Key {0} pressed".format(key))


def screenshot():
    # 获取屏幕截图
    screenshot = ImageGrab.grab()


def camera():
    # 打开摄像头
    ...


def execute_instructions(): ...


def kb_listener(signal: mp.Value) -> None:
    listener = pynput.keyboard.Listener(on_press=on_press_wrapper(signal))
    listener.start()
    listener.join()


if __name__ == "__main__":
    chat_comp = qianfan.ChatCompletion()
    msgs = qianfan.Messages()

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
    audio_collector = []

    signal = mp.Value("i", 0)
    listener = mp.Process(target=kb_listener, args=(signal,))
    listener.daemon = True
    print("按下F开始录音，按下S暂停录音，按下ESC退出程序")
    print("ready")
    listener.start()

    while True:
        if signal.value == 1:
            audio_data = stream.read(CHUNK)
            audio_collector.append(audio_data)
        elif signal.value == 0 and len(audio_collector) > 0:
            audio_array = np.frombuffer(b"".join(audio_collector), dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

            # segments, info = stt_model.transcribe(
            #     audio_array,
            #     language=args.record_src_language,
            #     beam_size=5,
            #     log_prob_threshold=-0.5,
            # )
            segments = pipe(audio_array, generate_kwargs=generate_kwargs)


            text = ""
            # for segment in segments:
            #     if segment.text != "":
            #         text += segment.text
            if segments['text'] != "":
                text += segments['text']
            print("[master]:", text)
            msgs.append(text)
            resp = chat_comp.do(model=args.model, messages=msgs)
            print("[assistant]:", resp["result"])
            if args.is_speak:
                pyttsx3.speak(resp["result"])
            msgs.append(resp)
            audio_collector = []

        elif signal.value == -1:
            break
        else:
            pass
        # time.sleep(0.2)

    listener.join()

    stream.stop_stream()
    stream.close()
    p.terminate()
