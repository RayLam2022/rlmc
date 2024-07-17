"""
@File    :   assistant.py
@Time    :   2024/07/04 12:21:43
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import os.path as osp
import time
import copy
import argparse

import cv2
import pyaudio
import pyttsx3
import numpy as np
import torch
from PIL import ImageGrab
from faster_whisper import WhisperModel

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

args = parser.parse_args()

os.environ["QIANFAN_ACCESS_KEY"] = args.access_key
os.environ["QIANFAN_SECRET_KEY"] = args.secret_key

SECONDS = 2
INT16_MAX_ABS_VALUE = 32768.0
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

model_size = "large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_size, device=device, compute_type="float16")


def speech_to_text():

    # 创建一个PyAudio对象
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
    counter = 0
    unit = RATE / CHUNK

    while True:
        audio_data = stream.read(CHUNK)
        audio_collector.append(audio_data)
        counter += 1
        if counter % (int(unit * SECONDS)+1) == 0:
            # 将音频数据转换为numpy数组
            audio_collector = np.frombuffer(b"".join(audio_collector), dtype=np.int16)
            audio_collector = audio_collector.astype(np.float32) / INT16_MAX_ABS_VALUE

            segments, info = model.transcribe(
                audio_collector, language="zh", beam_size=5, log_prob_threshold=-0.5
            )

            # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

            for segment in segments:
                # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                if segment.text != "":
                    print("[master]:", segment.text)

                msg = yield segment.text
                if msg != "":
                    pyttsx3.speak(msg)
                    # stream.write(data)
                    print("[assistant]:", msg)

            # initial
            audio_collector = []
            counter = 0

    # 关闭流和PyAudio对象
    stream.stop_stream()
    stream.close()
    p.terminate()


def screenshot():
    # 获取屏幕截图
    screenshot = ImageGrab.grab()


def camera():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()

        # 显示画面
        cv2.imshow("Camera", frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()


def call_qianfan():  # 待测试
    chat_comp = qianfan.ChatCompletion()
    msgs = qianfan.Messages()
    msg = ""
    while True:
        if msg != "":
            msgs.append(msg)
            resp = chat_comp.do(model=args.model, messages=msgs)
            # print("assistant:", resp["result"])
            msgs.append(resp)
            msg= yield resp["result"]
        else:
            msg = yield ""



def execute_instructions(): ...


def runchat():
    person = speech_to_text()
    computer = call_qianfan()
    next(person)
    next(computer)
    msg = ""
    while True:
        msg = person.send(msg)
        msg = computer.send(msg)


if __name__ == "__main__":
    runchat()
