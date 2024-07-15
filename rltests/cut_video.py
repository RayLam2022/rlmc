# ref: https://blog.51cto.com/u_16213369/9371852

import argparse
import os
import time

import av
import cv2


parser = argparse.ArgumentParser("cut video")

parser.add_argument(
    "-i", "--input_file", default="", required=True, help="video file", type=str
)
parser.add_argument(
    "-o", "--output_file", default="./output.mp4", type=str, help="output file"
)
parser.add_argument("-s", "--start", default=0.0, type=float, help="start second")
parser.add_argument("-e", "--end", default=-1.0, type=float, help="end second")
parser.add_argument(
    "-b",
    "--bbox",
    default=[],
    nargs="+",
    type=int,
    help="if bbox='', then cut whole video, else cut the bbox, bbox format int: x y w h",
)

args = parser.parse_args()


def main():
    # 打开音视频文件
    container = av.open(args.input_file)

    cap = cv2.VideoCapture(args.input_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration = total_frames / fps
    cap.release()

    print(
        f"total frames: {total_frames}, fps: {fps}, frame width: {frame_width}, frame height: {frame_height}, duration: {duration}"
    )

    # 获取音视频流
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0]

    layout = "stereo" if audio_stream.codec_context.channels > 1 else "mono"
    audio_sample_fmt = audio_stream.format.name
    audio_sample_rate = audio_stream.codec_context.sample_rate

    # 创建输出容器
    output_container = av.open(args.output_file, "w")

    # 添加音视频流到输出容器
    output_video_stream = output_container.add_stream(
        "h264", video_stream.codec_context.rate
    )
    output_audio_stream = output_container.add_stream(
        "aac", audio_stream.codec_context.rate
    )

    if args.bbox:
        x, y, w, h = args.bbox
        if (
            x + w <= frame_width
            and y + h <= frame_height
            and x < frame_width
            and y < frame_height
            and x >= 0
            and y >= 0
        ):
            output_video_stream.width = w
            output_video_stream.height = h
        else:
            raise ValueError("bbox out of range")

    # 设置剪辑的起始时间和结束时间
    start_time = args.start
    end_time = args.end

    for packet in container.demux():
        for frame in packet.decode():
            if packet.stream == video_stream:
                # 如果在剪辑的时间范围内，则编码并写入输出容器
                img_ndarray = frame.to_rgb().to_ndarray()

                if args.bbox:
                    img_ndarray = img_ndarray[y : y + h, x : x + w, :]

                new_frame = av.VideoFrame.from_ndarray(img_ndarray, format="rgb24")

                if end_time != -1.0:
                    if frame.time >= start_time and frame.time <= end_time:
                        output_packet = output_video_stream.encode(new_frame)
                        output_container.mux(output_packet)
                else:
                    if frame.time >= start_time:
                        output_packet = output_video_stream.encode(new_frame)
                        output_container.mux(output_packet)

            elif packet.stream == audio_stream:
                # 处理音频帧
                # 如果在剪辑的时间范围内，则编码并写入输出容器
                new_frame = av.AudioFrame.from_ndarray(
                    frame.to_ndarray(), format=audio_sample_fmt, layout=layout
                )
                new_frame.sample_rate = audio_sample_rate
                if end_time != -1.0:
                    if frame.time >= start_time and frame.time <= end_time:
                        output_packet = output_audio_stream.encode(new_frame)
                        output_container.mux(output_packet)
                else:
                    if frame.time >= start_time:
                        output_packet = output_audio_stream.encode(new_frame)
                        output_container.mux(output_packet)

    output_container.close()


if __name__ == "__main__":
    main()
