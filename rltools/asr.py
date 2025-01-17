"""
@File    :   stt.py
@Time    :   2024/07/10 09:57:46
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from typing import Optional, Union, List, Dict
import argparse
import time

import numpy as np
import torch
import pyaudio
import pynput
import pyperclip
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


parser = argparse.ArgumentParser("asr")
parser.add_argument(
    "-r",
    "--record_time",
    default=5,
    type=int,
    help="record time(second)",
)
parser.add_argument(
    "-l",
    "--record_src_language",
    default="zh",
    help="record src language",
)

parser.add_argument("-i", "--is_translate", action="store_true", help="is_translate")
parser.add_argument(
    "-s",
    "--translate_src_lang",
    default="zho_Hans",
    help="translate zho_Hans, zho_Hant,eng_Latn, jpn_Jpan,yue_Hant,kor_Hang",
)
parser.add_argument(
    "-t",
    "--translate_tgt_lang",
    default="eng_Latn",
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
INT16_MAX_ABS_VALUE = 32768.0
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


model_size = "large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = WhisperModel(model_size, device=device, compute_type="float16")


def on_press(key: pynput.keyboard.Key) -> Optional[bool]:
    if key == pynput.keyboard.Key.esc:
        global signal
        signal = 1
        return False
    elif hasattr(key, "char"):
        if key.char == "f":
            return False
    # print("Key {0} pressed".format(key))


def kb_listener() -> None:
    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()


def speech_to_text() -> str:
    # 创建一个PyAudio对象
    p = pyaudio.PyAudio()

    # 打开麦克风流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK,
    )

    audio_collector = []
    counter = 0
    unit = RATE // CHUNK
    text = ""
    start = time.time()
    print("start record...")
    while True:
        audio_data = stream.read(CHUNK)
        audio_collector.append(audio_data)
        counter += 1
        if counter % (int(unit * args.record_time) + 1) == 0:
            audio_collector = np.frombuffer(b"".join(audio_collector), dtype=np.int16)
            audio_collector = audio_collector.astype(np.float32) / INT16_MAX_ABS_VALUE
            print("record time: ", time.time() - start)
            segments, info = stt_model.transcribe(
                audio_collector,
                language=args.record_src_language,
                beam_size=5,
                log_prob_threshold=-0.5,
            )

            text = ""
            for segment in segments:
                # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                if segment.text != "":
                    text += segment.text
            # initial
            audio_collector = []
            counter = 0
            break

    # 关闭流和PyAudio对象
    stream.stop_stream()
    stream.close()
    p.terminate()

    return text


def main() -> None:
    global signal

    print("record_lang: ", args.record_src_language)

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

    while True:
        kb_listener()
        if signal == 1:
            break
        text = speech_to_text()
        print("record is done")
        translated_text = ""
        if args.is_translate and text != "":
            for tl in translator(text):
                translated_text += tl["translation_text"]
            print("sequence is translated")
            text = translated_text
        pyperclip.copy(text)


if __name__ == "__main__":
    main()
