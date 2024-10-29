from typing import Optional, Union, List, Dict
import argparse
import datetime
import subprocess
from pathlib import Path

import rich
import numpy as np
import torch
import torchaudio
import srt
from tqdm import tqdm

# from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


parser = argparse.ArgumentParser("asr")

parser.add_argument(
    "-f",
    "--video_files",
    required=True,
    help="video files or audio files, mp4,mp3...",
    nargs="+",
)

parser.add_argument(
    "-l",
    "--video_src_language",
    default="zh",
    required=True,
    help="video src language",
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


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0



model_id = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# stt_model = WhisperModel(model_size, device=device, compute_type="float16")
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
stt_model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)

generate_kwargs = {
    "language": args.video_src_language,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

rprint=rich.console.Console().print

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
    rprint(
        "translate model is ready, src_lang: ",
        args.translate_src_lang,
        "tgt_lang: ",
        args.translate_tgt_lang,
    )

def main() -> None:
    for video_file in args.video_files:
        rprint("processing %s" % video_file, style="bold green")

        # waveform, sample_rate = torchaudio.load(video_file)
        # audio=waveform[0].numpy()
        #waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio= load_audio(video_file, sr=16000)

        video_file = Path(video_file)
        output_file = video_file.parent / (video_file.stem + ".srt")

        segments = pipe(
            audio, generate_kwargs=generate_kwargs, chunk_length_s=10, return_timestamps=True
        )

        index=1
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in segments["chunks"]:
                if args.is_translate:
                    text=segment['text']
                    translated_text=""
                    for tl in translator(text, max_length=1000):
                        translated_text+=tl["translation_text"]
                    text+= '\n' + translated_text
                else:
                    text=segment['text']
                words=srt.Subtitle(index, datetime.timedelta(seconds=segment['timestamp'][0]), datetime.timedelta(seconds=segment['timestamp'][1]), text)
                f.writelines(words.to_srt())
                index+=1
    rprint('Done',style="bold green")

if __name__ == "__main__":
    main()



