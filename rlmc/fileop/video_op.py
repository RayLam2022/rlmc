"""
@File    :   video_op.py
@Time    :   2024/07/03 08:59:38
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import os
from typing import Literal

import av
import cv2
import numpy as np

from rlmc.fileop.abstract_file import AbstractFile

__all__ = ["VideoFile"]


# 人脸检测器
class FaceDetector:
    def __init__(self, module_file):
        self.module_file = module_file
        self.face_cascade = cv2.CascadeClassifier(self.module_file)

    def detectFace(self, gray_img):
        face_rect = self.face_cascade.detectMultiScale(gray_img, 1.3, 5)
        return face_rect


class VideoFile:
    def __init__(
        self,
        file_path: str = "",
        # mode: Literal["stream", "camera", "file"] = "file",
        vcodec: Literal["mpeg4", "libx264"] = "libx264",
        pixformat: Literal["rgb24"] = "rgb24",
        width: int = 0,
        height: int = 0,
        fps: int = 30,
        duration: float = 0.0,
        acodec: Literal["aac", "mp3"] = "aac",  #
        bit_rate: int = 2000000,
        audio_channels: int = 1,
        audio_sample_rate: int = 44100,
        audio_sample_fmt: str = "",
        is_save: bool = False,
        save_path: str = "",
        tread_type: Literal["SLICE", "AUTO"] = "SLICE",
    ) -> None:

        self.container = None
        self.output_container = None
        self.file_path = file_path
        self.vcodec = vcodec
        self.pixformat = pixformat
        self.acodec = acodec
        self.bit_rate = bit_rate
        self.audio_channels = audio_channels
        self.audio_sample_rate = audio_sample_rate
        self.audio_sample_fmt = audio_sample_fmt
        self.audio_format_dtypes = {
            "dbl": "<f8",
            "dblp": "<f8",
            "flt": "<f4",
            "fltp": "<f4",
            "s16": "<i2",
            "s16p": "<i2",
            "s32": "<i4",
            "s32p": "<i4",
            "u8": "u1",
            "u8p": "u1",
        }
        self.is_save = is_save
        self.save_path = save_path
        self.fps = fps
        self.duration = duration
        self.total_frames = 0
        self.tread_type = tread_type
        self.width = width
        self.height = height

        if file_path != "":
            self.container = self.read()

    def __enter__(self):
        self.read()
        if self.is_save:
            self.output_container = av.open(self.save_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.close()
        if self.is_save:
            self.output_container.close()
        if exc_type != None:
            print(exc_type, exc_val, exc_tb)

    # def __iter__(self):
    #     self.read()
    #     return self.container.demux()

    # def __next__(self):
    #     return self.file.__next__()

    def add_vid_stream(self):
        output_stream = self.output_container.add_stream(self.vcodec, rate=self.fps)
        codec = output_stream.codec_context
        codec.width = self.width
        codec.height = self.height
        codec.pix_fmt = self.pixformat
        codec.bit_rate = self.bit_rate
        return output_stream

    def add_aud_stream(self):
        output_stream = self.output_container.add_stream(
            self.acodec, rate=self.audio_sample_rate
        )
        return output_stream

    def read(self):
        cap = cv2.VideoCapture(self.file_path)
        self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        cap.release()
        self.container = av.open(self.file_path)
        self.bit_rate = self.container.streams.video[0].codec_context.bit_rate
        self.pixformat = self.container.streams.video[0].codec_context.pix_fmt
        self.width = self.container.streams.video[0].codec_context.width
        self.height = self.container.streams.video[0].codec_context.height
        self.audio_sample_rate = self.container.streams.audio[
            0
        ].codec_context.sample_rate
        self.audio_channels = self.container.streams.audio[0].codec_context.channels
        self.audio_sample_fmt = self.container.streams.audio[0].format.name
        return self.container


if __name__ == "__main__":
    module_file = r"D:\RLdownload\haarcascade_frontalface_alt2.xml"
    src_video = r"C:\Users\RayLam\Desktop\ai_design\cctv.mp4"
    is_show = True
    is_save = True
    save_path = r"C:\Users\RayLam\Desktop\cctv_new.mp4"

    dectector = FaceDetector(module_file)

    with VideoFile(
        src_video,
        is_save=is_save,
        save_path=save_path,
    ) as f:

        # container = f.read()
        out_vid_stream = f.add_vid_stream()
        out_aud_stream = f.add_aud_stream()
        flag = False
        for packet in f.container.demux():
            if packet.stream.type == "audio":
                for frame in packet.decode():
                    format_dtype = np.dtype(f.audio_format_dtypes[f.audio_sample_fmt])
                    layout = "stereo" if f.audio_channels > 1 else "mono"
                    print("index:", frame.index)
                    print("时间戳:", frame.pts)
                    print("channels:", f.audio_channels)
                    print("采样率:", frame.sample_rate)
                    print("采样格式:", f.audio_sample_fmt)
                    print(frame.to_ndarray().shape)
                    new_frame = av.AudioFrame.from_ndarray(
                        frame.to_ndarray(), format=f.audio_sample_fmt, layout=layout
                    )
                    new_frame.sample_rate = f.audio_sample_rate
                    for packet in out_aud_stream.encode(new_frame):
                        f.output_container.mux(packet)

            elif packet.stream.type == "video":
                for frame in packet.decode():
                    print("index:", frame.index)
                    print("时间戳:", frame.pts)
                    print("帧类型:", frame.pict_type)
                    print("宽度:", frame.width)
                    print("高度:", frame.height)
                    img_ndarray = frame.to_rgb().to_ndarray()
                    print("像素数据:", img_ndarray.shape)

                    # 检测人脸
                    gray = cv2.cvtColor(img_ndarray, cv2.COLOR_RGB2GRAY)
                    faces = dectector.detectFace(gray)
                    for x, y, w, h in faces:
                        cv2.rectangle(
                            img_ndarray, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )

                    new_frame = av.VideoFrame.from_ndarray(img_ndarray, format="rgb24")
                    for packet in out_vid_stream.encode(new_frame):
                        f.output_container.mux(packet)

                    if is_show:
                        img_ndarray = cv2.cvtColor(img_ndarray, cv2.COLOR_RGB2BGR)
                        cv2.imshow("frame", img_ndarray)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            flag = True
                            break
            if flag:
                break
        # flush stream
        for packet in out_vid_stream.encode():
            f.output_container.mux(packet)
        for packet in out_aud_stream.encode():
            f.output_container.mux(packet)
        if is_show:
            cv2.destroyAllWindows()
