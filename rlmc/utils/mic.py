"""
@File    :   mic.py
@Time    :   2024/07/04 12:21:43
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import multiprocessing as mp
import os
import os.path as osp
import time
import copy
import argparse

import pynput
import pyaudio
import wave
import numpy as np


parser = argparse.ArgumentParser("microphone")
parser.add_argument(
    "-d",
    "--save_dir",
    required=True,
)
args = parser.parse_args()

INT16_MAX_ABS_VALUE = 32768.0
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
unit = RATE // CHUNK


def on_press_wrapper(signal):
    def on_press(key):
        nonlocal signal
        if key == pynput.keyboard.Key.esc:
            signal.value = -1
            return False
        elif hasattr(key, "char"):
            if key.char == "f":
                print("start record...")
                signal.value = 1
            if key.char == "s":
                print("end record...")
                signal.value = 0

    return on_press
    # print("Key {0} pressed".format(key))


def kb_listener(signal):
    listener = pynput.keyboard.Listener(on_press=on_press_wrapper(signal))
    listener.start()
    listener.join()


if __name__ == "__main__":

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
    daytime_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    print("按下F开始录音，按下S暂停录音，按下ESC退出程序")
    print("ready")
    listener.start()

    while True:
        if signal.value == 1:
            audio_data = stream.read(CHUNK)
            audio_collector.append(audio_data)

        elif signal.value == -1:
            break
        else:
            pass
        # time.sleep(0.2)
    # audio_array = np.frombuffer(b"".join(audio_collector), dtype=np.int16)
    # audio_array = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

    listener.join()

    stream.stop_stream()
    stream.close()
    p.terminate()

    daytime_end = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    wf = wave.open(
        os.path.join(args.save_dir, f"{daytime_start}__{daytime_end}.wav"), "wb"
    )
    # 设置音频的声道数、采样宽度和采样率
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    # 将 frames 列表中的所有数据连接成一个字节串，并写入文件
    wf.writeframes(b"".join(audio_collector))
    # 关闭文件
    wf.close()
