import sys

if "." not in sys.path:
    sys.path.append(".")

from typing import Dict

import rlmc
import os

import shutil


def iter_files(rootDir, keyword_a="", keyword_b=""):
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root, file)
            if keyword_a in file_name and keyword_b in file_name:
                os.remove(file_name)
                print(file_name)

        for di in dirs:
            di_name = os.path.join(root, di)
            if keyword_a in di_name and keyword_b in di_name:
                shutil.rmtree(di_name)
                print(di_name)


def ipyclear(
    rootDir=r"/root/Pyramid-Attention-Networks/DIV2K", keyword=".ipynb_checkpoints"
):
    iter_files(rootDir, keyword)


if __name__ == "__main__":
    rootDir = r"D:\work\rlmc"
    keyword = ".ipynb_checkpoints"
    ipyclear(rootDir, keyword)
