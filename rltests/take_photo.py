"""
@File    :   take_photo.py
@Time    :   2024/07/20 21:22:38
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import argparse
import os
import os.path as osp

import cv2
import rich
from rich.console import Console

parser = argparse.ArgumentParser("take photo")
parser.add_argument("-o", "--output_dir", type=str, required=True)
parser.add_argument("-w", "--width", type=int, default=0)
parser.add_argument("-ht", "--height", type=int, default=0)
args = parser.parse_args()

rc = Console()


def main():

    rc.print(
        "[green]###  f: 拍摄图片",
    )  # style=rich.style.Style(color='white', bgcolor='white')
    rc.print("[green]###  esc: 退出")

    cap = cv2.VideoCapture(
        0,
    )  # cv2.CAP_DSHOW
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rc.print(f"[green]###  摄像头分辨率 w x h: {width} x {height}")
    rc.rule("[bold green]Take Photo")
    index = 0
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1, dst=None)
        if args.width > 0 and args.height > 0:
            frame = cv2.resize(
                frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR
            )
        cv2.imshow("capture", frame)
        input = cv2.waitKey(1) & 0xFF

        if input == ord("f"):
            cv2.imwrite("%s/%d.jpg" % (args.output_dir, index), frame)
            print("%s: %d 张图片" % (args.output_dir, index))
            index += 1
        elif input == 27:
            break
        else:
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
