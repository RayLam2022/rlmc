"""
@File    :   camera.py
@Time    :   2024/07/04 00:00:14
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

# demo not be used in the project

import multiprocessing as mp
from os import remove, mkdir, listdir
from os.path import exists, splitext, basename, join
from datetime import datetime
from time import sleep
import time
from fractions import Fraction

import pyaudio
from PIL import ImageGrab
from numpy import array
import wave
import cv2
import av
import numpy as np


MODE = "merge"  # 'screen' or 'camera' or 'merge'
IS_DISPLAY = True
CHUNK_sIZE = 4096
CHANNELS = 1
FORMAT = pyaudio.paFloat32  # pyaudio.paInt16
RATE = 48000


def record_audio(mgr_dict, is_display=False):
    p = pyaudio.PyAudio()
    # event.wait()
    sleep(3)
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        # input_device_index=4,#立体混音，具体选哪个根据需要选择
        output=True,
        frames_per_buffer=CHUNK_sIZE,
    )

    while mgr_dict["allowRecording"]:
        # 从录音设备读取数据，直接写入wav文件
        data = stream.read(CHUNK_sIZE)
        if is_display:
            stream.write(data)
        np_data = np.frombuffer(data, dtype=np.float32)
        mgr_dict["audio_data"] = (np_data)


    stream.stop_stream()
    stream.close()
    p.terminate()


def record_screen(mgr_dict, is_display=False):
    # 录制屏幕
    im = ImageGrab.grab()
    while mgr_dict["allowRecording"]:
        im = ImageGrab.grab()
        im = cv2.cvtColor(array(im), cv2.COLOR_RGB2BGR)
        if is_display:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imshow("im", im)
            cv2.waitKey(1)
    cv2.destroyAllWindows()


def record_cam(mgr_dict, is_display=False):
    # 录制屏幕
    cap = cv2.VideoCapture(0)  # camera id
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened() and mgr_dict["allowRecording"]:
        ret, im = cap.read()
        if ret:
            if is_display:
                cv2.imshow("im", im)
                cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def merge(mgr_dict, output_path, is_display=False):
    sleep(3)
    
    start=time.time()
    cap = cv2.VideoCapture(0)  # camera id
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_base=Fraction(1, int(fps))
    container = av.open(output_path, mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = frame_width
    stream.height = frame_height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.time_base = time_base

    audio_stream = container.add_stream("mp3", rate=RATE)
    audio_stream.codec_context.time_base = time_base
    layout = "stereo" if CHANNELS > 1 else "mono"
    log_ndarray = np.array(1.0)
    while cap.isOpened() and mgr_dict["allowRecording"]:
        ret, im = cap.read()
        if ret:
            if is_display:
                cv2.imshow("im", im)
                cv2.waitKey(1)
            im_frame = av.VideoFrame.from_ndarray(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24"
            )
            for packet in stream.encode(im_frame):
                container.mux(packet)
            if int((time.time()-start)*1000) % 2 ==0 :continue ######隔接收一次音频数据
            np_data = mgr_dict.get("audio_data")
            if not isinstance(np_data, type(None)):
                if log_ndarray.sum()==np_data.sum(): continue

                if np_data.ndim == 1:
                    np_data = np.reshape(np_data, (1, -1))
                new_frame = av.AudioFrame.from_ndarray(
                    np_data, format="fltp", layout=layout
                )

                new_frame.sample_rate = RATE

                for packet in audio_stream.encode(new_frame):
                    container.mux(packet)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


root = r"C:\Users\RayLam\Desktop\temp_file"
now = str(datetime.now())[:19].replace(":", "_")
video_filename = join(root, "%s.mp4" % now)


if __name__ == "__main__":
    mgr_dict = mp.Manager().dict()
    mgr_dict["allowRecording"] = True
    

    t1 = mp.Process(target=record_audio, args=(mgr_dict, IS_DISPLAY))
    if MODE == "screen":
        t2 = mp.Process(target=record_screen, args=(mgr_dict, IS_DISPLAY))
    elif MODE == "camera":
        t2 = mp.Process(target=record_cam, args=(mgr_dict, IS_DISPLAY))
    else:
        t2 = mp.Process(target=merge, args=(mgr_dict, video_filename, IS_DISPLAY))

    for t in (t1, t2):
        t.daemon=True

    for t in (t1, t2):
        t.start()

    print("3秒后开始录制，按q键回车结束录制")
    while True:
        if input() == "q":
            break
    mgr_dict["allowRecording"] = False
    for i in (t1, t2):
        t.join()

    print("End")
