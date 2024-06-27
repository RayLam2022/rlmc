"""
@File    :   utils.py
@Time    :   2024/06/25 10:36:13
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import os
import os.path as osp
import glob
import shutil
import random
import re

import numpy as np
from tqdm import tqdm

__all__ = ["DatasetSplit", "FindContent"]


class DatasetSplit:
    """
    if __name__ == "__main__":
        ds = DatasetSplit(
            "./imgs",
            "./labels",
            img_exts=["*.jpg"],
            label_ext="*.xml",
            split={"train": 0.8, "val": 0.1, "test": 0.1},
            is_split_name_up=False,
        )
        ds.process()
    """

    def __init__(
        self,
        img_dir,
        label_dir,
        img_exts=["*.jpg"],
        label_ext="*.png",
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        is_shuffle=True,
        random_seed=123,
        is_split_name_up=True,
    ):
        if sum(split.values()) != 1:
            raise ValueError("split values must sum to 1")
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = self._collect_files(img_dir, img_exts)
        self.label_ext = label_ext.split(".")[-1]
        self.output_dir = osp.dirname(self.img_dir) + "/processed"
        self.split = split
        self.is_split_name_up = is_split_name_up
        self.pairs = self._pair_files()
        print(self.pairs)
        if is_shuffle:
            random.seed(random_seed)
            random.shuffle(self.pairs)

    def _collect_files(self, directory, exts=["*.jpg", "*.png"]):
        files = []
        for ext in exts:
            files.extend(glob.glob(osp.join(directory, ext)))
        return files

    def _pair_files(self):
        pairs = []
        for img_file in self.img_files:
            label_file = osp.join(
                self.label_dir,
                osp.splitext(osp.basename(img_file))[0] + "." + self.label_ext,
            )
            if osp.isfile(label_file):
                pairs.append((img_file, label_file))
        return pairs

    def _split_files(self):
        img_lab = ["images", "labels"]  # 可改此处改目录文件夹名

        lenth = len(self.pairs)
        counter = 0

        for split_idx, (k, v) in enumerate(self.split.items()):
            print("################### processing: ", k, " #####################")
            num = int(lenth * v)
            if split_idx == len(self.split) - 1:
                num = lenth - counter
            assert num > 1, f"{k}数据过少，请检查数据集"

            for i in tqdm(range(num)):
                img_file, label_file = self.pairs[counter]
                if self.is_split_name_up:
                    img_tgt = osp.join(
                        self.output_dir, k, img_lab[0], osp.basename(img_file)
                    )
                    label_tgt = osp.join(
                        self.output_dir, k, img_lab[1], osp.basename(label_file)
                    )
                else:
                    img_tgt = osp.join(
                        self.output_dir, img_lab[0], k, osp.basename(img_file)
                    )
                    label_tgt = osp.join(
                        self.output_dir, img_lab[1], k, osp.basename(label_file)
                    )
                os.makedirs(osp.dirname(img_tgt), exist_ok=True)
                os.makedirs(osp.dirname(label_tgt), exist_ok=True)

                shutil.copy(img_file, img_tgt)
                shutil.copy(label_file, label_tgt)
                counter += 1
        print("counter:", counter, "  data_total_lenth:", lenth)

    def process(self):
        self._split_files()
        print("Done!")


class FindContent:
    """
    if __name__ == "__main__":
        fc=FindContent(
            content=" $waffa",
            file_dir="./nginx1-7-11-3Gryphon",
            exts=['*.conf','*.py', '*.exe'], encoding='utf-8',is_use_regex=False, is_recursive=True
        )
        res=fc.process()
        print(res)
    """

    def __init__(
        self,
        content,
        file_dir,
        exts=["*.conf", "*.py"],
        encoding="utf-8",
        is_use_regex=False,
        is_recursive=True,
    ):
        self.content = content
        self.file_dir = file_dir
        self.exts = exts
        self.encoding = encoding
        self.is_use_regex = is_use_regex
        self.is_recursive = is_recursive

    def find_string_in_file(self, file_path, search_string):
        with open(file_path, "r", encoding=self.encoding) as file:
            for line in file.readlines():
                if self.is_use_regex:
                    if re.search(search_string, line):
                        return True
                else:
                    if search_string in line:
                        return True
        return False

    def process(self):
        files = []
        for ext in self.exts:
            if self.is_recursive:
                files.extend(
                    glob.glob(
                        osp.join(self.file_dir, "**", ext), recursive=self.is_recursive
                    )
                )
            else:
                files.extend(
                    glob.glob(osp.join(self.file_dir, ext), recursive=self.is_recursive)
                )
        print(files)
        collector = []
        for file in tqdm(files):
            try:
                is_found = self.find_string_in_file(file, self.content)
            except UnicodeDecodeError as ue:
                print(f"\nskip:UnicodeDecodeError processing file {file}: {ue}")
                is_found = False
            except Exception as e:
                print(f"\nskip:Error processing file {file}: {e}")
                is_found = False
            if is_found:
                collector.append(file)

        return collector


if __name__ == "__main__":
    # ds = DatasetSplit(
    #     "./imgs",
    #     "./labels",
    #     img_exts=["*.jpg"],
    #     label_ext="*.xml",
    #     split={"train": 0.8, "val": 0.1, "test": 0.1},
    #     is_split_name_up=False,
    # )
    # ds.process()

    fc = FindContent(
        content=".*waffable.*",  # ".* waffa.*",
        file_dir=r"",
        exts=["*.conf", "*.py", "*.exe"],
        encoding="utf-8",
        is_use_regex=True,
        is_recursive=False,
    )
    res = fc.process()
    print(res)
