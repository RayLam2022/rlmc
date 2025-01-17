import argparse
from pathlib import Path

from moviepy.editor import VideoFileClip


parser = argparse.ArgumentParser("cut video")

parser.add_argument(
    "-i", "--input_file", default="", required=True, help="video file", type=str
)

parser.add_argument("-s", "--start", default="00:00:00", type=str, help="start time '00:00:00' ")
parser.add_argument("-e", "--end", default="-1.0", type=str, help="end time '00:00:00', -1.0 means end of the video")
parser.add_argument(
    "-b",
    "--bbox",
    default=[],
    nargs="+",
    type=int,
    help="if bbox=[], then cut whole video, else cut the bbox, bbox format int: x y w h",
)

args = parser.parse_args()


codec={'.mp4':('libx264', 'aac'),
       '.avi':('libxvid', 'mp3'),}


def main() -> None:
    input_file=Path(args.input_file)
    vid = VideoFileClip(str(input_file), audio=True)
    print(f"duration: {vid.duration}  width: {vid.w}   height: {vid.h}")

    if args.end == "-1.0":
        end = vid.duration
    else:
        end = args.end
    start = args.start

    clip = vid.subclip(start, end)

    if args.bbox:
        x, y, w, h = args.bbox
        if (
            x + w <= vid.w
            and y + h <= vid.h
            and x < vid.w
            and y < vid.h
            and x >= 0
            and y >= 0
        ):
            clip = clip.crop(x1=int(x), y1=int(y), x2=int(x + w), y2=int(y + h))
        else:
            raise ValueError("bbox out of range")
    output_file=input_file.parent.joinpath(input_file.stem + "_cut" + input_file.suffix)

    clip.write_videofile(str(output_file),codec=codec[input_file.suffix][0] ,audio_codec=codec[input_file.suffix][1])
    vid.close()
    print("Done!")


if __name__ == "__main__":
    main()
