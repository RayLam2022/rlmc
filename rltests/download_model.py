import sys

if "." not in sys.path:
    sys.path.append(".")

import argparse
import os

from rlmc.utils.downloadscript import HfDownload, MsDownload, AutodlDownload


parser = argparse.ArgumentParser("download model")
parser.add_argument("-m", "--method", default="hf", help="hf,ms,autodl")
parser.add_argument("-r", "--repo_id", required=True, help="repo id")
parser.add_argument("-x", "--XDG_CACHE_HOME", default="", help="XDG_CACHE_HOME")
parser.add_argument("-c", "--cache_dir", default="", help="cache dir")
parser.add_argument("-l", "--is_login", default=False, help="is login hf")
parser.add_argument("-t", "--hf_token", default="", help="hf token")
parser.add_argument("-i", "--ignore_patterns", default=[], nargs="+", help="*.h5  *safetensors  *msgpack")

args = parser.parse_args()

if args.XDG_CACHE_HOME:
    os.environ["XDG_CACHE_HOME"] = args.XDG_CACHE_HOME
    os.environ["MODELSCOPE_CACHE"] = os.path.join(args.XDG_CACHE_HOME, "modelscope")


def main():
    print("sys XDG_CACHE_HOME: ", os.environ.get("XDG_CACHE_HOME"))
    if args.method == "hf":
        dl = HfDownload(
            args.repo_id, args.cache_dir, ignore_patterns=args.ignore_patterns, hf_token=args.hf_token, is_login=args.is_login
        )
    elif args.method == "ms":
        dl = MsDownload(args.repo_id, args.cache_dir)
    else:
        dl = AutodlDownload(args.repo_id)
    dl.run()
