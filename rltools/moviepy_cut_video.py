import argparse

from moviepy.editor import VideoFileClip


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

    vid = VideoFileClip(args.input_file, audio=True)
    print(f"duration: {vid.duration}  width: {vid.w}   height: {vid.h}")

    if args.end == -1.0:
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
    clip.write_videofile(args.output_file, audio_codec="aac")
    vid.close()
    print("Done!")


if __name__ == "__main__":
    main()
