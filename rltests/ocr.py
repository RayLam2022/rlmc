"""
@File    :   ocr.py
@Time    :   2024/07/09 12:05:48
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from importlib import import_module
import multiprocessing as mp
import os
import sys
import time
import io
import argparse

import cv2
from PIL import ImageGrab
import numpy as np
import pynput
import pyperclip
import easyocr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


parser = argparse.ArgumentParser("ocr")
parser.add_argument(
    "-l",
    "--language",
    nargs="+",
    default=["ch_sim", "en"],
    help="ch_sim, en, ko, ja and so on",
)
parser.add_argument(
    "-tl", "--tolerance", default=0.30, help="the tolerance to divide the line"
)
parser.add_argument("-i", "--is_translate", action="store_true", help="is_translate")
parser.add_argument(
    "-s",
    "--translate_src_lang",
    default="eng_Latn",
    help="translate zho_Hans, zho_Hant,eng_Latn, jpn_Jpan,yue_Hant,kor_Hang",
)
parser.add_argument(
    "-t",
    "--translate_tgt_lang",
    default="zho_Hans",
    help="translate zho_Hans, zho_Hant,eng_Latn, jpn_Jpan,yue_Hant,kor_Hang",  # zho_Hant繁体
)
parser.add_argument(
    "-m",
    "--translate_model",
    default="facebook/nllb-200-distilled-600M",
    help="translate model",
)
args = parser.parse_args()


signal = 0


def check_bbox(bbox):
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
        return False
    if bbox[0] > bbox[2] or bbox[1] >= bbox[3]:
        return False
    return True


def get_screenshot(bbox):
    byteImgIO = io.BytesIO()
    img = ImageGrab.grab(bbox)
    img.save(byteImgIO, format="PNG")
    byteImgIO.seek(0)
    img = cv2.imdecode(np.frombuffer(byteImgIO.read(), np.uint8), cv2.IMREAD_COLOR)
    return img


def on_click(x, y, button, pressed):
    global bbox
    if not pressed:
        bbox[2:] = [x, y]
        return False
    else:
        bbox[0:2] = [x, y]


def on_press(key):
    if key == pynput.keyboard.Key.esc:
        global signal
        signal = 1
        return False
    elif hasattr(key, "char"):
        if key.char == "f":
            return False

    # print("Key {0} pressed".format(key))


def ms_listener():
    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()
    listener.join()


def kb_listener():
    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()


def main():
    global signal
    global bbox
    print("lang: ", args.language)
    reader = easyocr.Reader(args.language)
    if args.is_translate:
        tokenizer = AutoTokenizer.from_pretrained(args.translate_model)
        translate_model = AutoModelForSeq2SeqLM.from_pretrained(args.translate_model)

        translator = pipeline(
            "translation",
            model=translate_model,
            tokenizer=tokenizer,
            src_lang=args.translate_src_lang,
            tgt_lang=args.translate_tgt_lang,
            max_length=512,
            device=0,
        )
        print(
            "translate model is ready, src_lang: ",
            args.translate_src_lang,
            "tgt_lang: ",
            args.translate_tgt_lang,
        )
    print('easeocr is ready, press "f" to select area, press "esc" to exit')
    while True:
        bbox = [-1, -1, -1, -1]
        kb_listener()
        if signal == 1:
            break
        ms_listener()
        # print(bbox)
        if check_bbox(bbox):
            img = get_screenshot(bbox)
            result = reader.readtext(img)
            print(result)
            text = ""
            translated_text = ""
            y_top_compare = 0
            y_bot_compare = 0
            for idx, res in enumerate(result):

                y_top_val, y_bot_val = result[idx][0][0][1], result[idx][0][2][1]
                gap = (y_bot_val - y_top_val) * args.tolerance
                # 如y_top_val大等于y_top_compare减gap 且 y_bot_compare加gap大等于y_bot_val则在同一行，用空格间隔，否则就用回车换行
                if y_top_compare == 0:
                    text += result[idx][1]
                elif (
                    y_top_val - (y_top_compare - gap) >= 0
                    and (y_bot_compare + gap) - y_bot_val >= 0
                ) :
                    text += " " + result[idx][1] 
                else:
                    text += "\n" + result[idx][1]

                if (
                    abs(y_bot_compare - y_top_compare) <= abs(y_bot_val - y_top_val)
                    or y_top_compare == 0
                ):
                    y_top_compare, y_bot_compare = y_top_val, y_bot_val
            if args.is_translate:
                for tl in translator(text):
                    translated_text += tl["translation_text"]
                print('sequence is translated')
                pyperclip.copy(text + "\n" + translated_text)
            else:
                pyperclip.copy(text)
        else:
            print("invalid bbox")


if __name__ == "__main__":
    main()
