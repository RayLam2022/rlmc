"""
@File    :   live.py
@Time    :   2024/07/04 00:00:14
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import multiprocessing as mp
from os import remove, mkdir, listdir
from os.path import exists, splitext, basename, join
from datetime import datetime
import time
from fractions import Fraction

import pyaudio
from PIL import ImageGrab
import cv2
import av
import numpy as np


MODE = "screen"  # 'screen' or 'camera'
IS_DISPLAY = False
IS_SAVE = True
CHUNK_SIZE = 4096
CHANNELS = 1
FORMAT = pyaudio.paFloat32  # pyaudio.paInt16
RATE = 48000


def record_audio(mgr_dict, mgr_audio_data):
    p = pyaudio.PyAudio()
    time.sleep(3)
    start = time.time()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        # input_device_index=4,#立体混音或其他
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    while mgr_dict["allowRecording"]:
        data = stream.read(CHUNK_SIZE)
        if IS_DISPLAY:
            stream.write(data)
        if IS_SAVE:
            np_data = np.frombuffer(data, dtype=np.float32)
            pts = int((time.time() - start) * RATE)
            mgr_audio_data[pts] = np_data

    stream.stop_stream()
    stream.close()
    p.terminate()


def record_screen(mgr_dict, mgr_audio_data, output_path):
    # 录制屏幕
    time.sleep(3)
    start = time.time()
    im = ImageGrab.grab()
    im = np.array(im)
    fps = 30
    h, w, c = im.shape

    if IS_SAVE:
        container = av.open(output_path, mode="w", format="mp4")
        stream = container.add_stream("h264", rate=fps) 
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = "yuv420p" # 'nv16'   #
        stream.bit_rate=2_000_000
        stream.codec_context.time_base = Fraction(1, int(fps))

        audio_stream = container.add_stream("mp3", rate=RATE)
        audio_stream.codec_context.time_base = Fraction(1, int(RATE))
        layout = "stereo" if CHANNELS > 1 else "mono"

    while mgr_dict["allowRecording"]:
        im = ImageGrab.grab()
        im = np.array(im)
        if IS_DISPLAY:
            cv2.imshow("im", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        if IS_SAVE:
            im_frame = av.VideoFrame.from_ndarray(im, format="rgb24")
            for packet in stream.encode(im_frame):
                container.mux(packet)

            np_data_dict = mgr_audio_data

            for pts, np_data in np_data_dict.items():
                if np_data.ndim == 1:
                    np_data = np.reshape(np_data, (1, -1))
                new_frame = av.AudioFrame.from_ndarray(
                    np_data, format="fltp", layout=layout
                )

                new_frame.sample_rate = RATE
                new_frame.pts = pts

                for packet in audio_stream.encode(new_frame):
                    container.mux(packet)

                mgr_audio_data.pop(pts)

    if IS_DISPLAY:
        cv2.destroyAllWindows()
    if IS_SAVE:
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

        # Close the file
        container.close()


def record_cam(mgr_dict, mgr_audio_data, output_path):
    time.sleep(3)
    start = time.time()

    cap = cv2.VideoCapture(0)  # camera id
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if IS_SAVE:
        container = av.open(output_path, mode="w",format="mp4")
        stream = container.add_stream("h264", rate=fps)
        stream.width = frame_width
        stream.height = frame_height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate=2_000_000
        stream.codec_context.time_base = Fraction(1, int(fps))

        audio_stream = container.add_stream("mp3", rate=RATE)
        audio_stream.codec_context.time_base = Fraction(1, int(RATE))
        layout = "stereo" if CHANNELS > 1 else "mono"

    while cap.isOpened() and mgr_dict["allowRecording"]:
        ret, im = cap.read()
        if ret:
            if IS_DISPLAY:
                cv2.imshow("im", im)
                cv2.waitKey(1)
            if IS_SAVE:
                im_frame = av.VideoFrame.from_ndarray(
                    cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24"
                )
                for packet in stream.encode(im_frame):
                    container.mux(packet)

                np_data_dict = mgr_audio_data

                for pts, np_data in np_data_dict.items():
                    if np_data.ndim == 1:
                        np_data = np.reshape(np_data, (1, -1))
                    new_frame = av.AudioFrame.from_ndarray(
                        np_data, format="fltp", layout=layout
                    )

                    new_frame.sample_rate = RATE
                    new_frame.pts = pts

                    for packet in audio_stream.encode(new_frame):
                        container.mux(packet)

                    mgr_audio_data.pop(pts)
        else:
            break

    cap.release()
    if IS_DISPLAY:
        cv2.destroyAllWindows()
    if IS_SAVE:
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

        # Close the file
        container.close()


root = r"C:\Users\\Desktop\temp_file"
now = str(datetime.now())[:19].replace(":", "_")
video_output = join(root, "%s.mp4" % now)


if __name__ == "__main__":
    mgr_dict = mp.Manager().dict()
    mgr_dict["allowRecording"] = True
    mgr_audio_data = mp.Manager().dict()

    t1 = mp.Process(target=record_audio, args=(mgr_dict, mgr_audio_data))
    if MODE == "screen":
        t2 = mp.Process(
            target=record_screen, args=(mgr_dict, mgr_audio_data, video_output)
        )
    elif MODE == "camera":
        t2 = mp.Process(
            target=record_cam, args=(mgr_dict, mgr_audio_data, video_output)
        )

    for t in (t1, t2):
        t.daemon = True

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
