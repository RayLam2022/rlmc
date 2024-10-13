import hashlib
import argparse

import rich.progress

parser = argparse.ArgumentParser("generate hash")
parser.add_argument("-f", "--file", required=True)
parser.add_argument("-a", "--algorithm", default="md5", help="md5, sha1, sha256")
args = parser.parse_args()


def encrypt(fpath: str, algorithm: str) -> str:
    with rich.progress.open(fpath, "rb") as f:
        hash = hashlib.new(algorithm)
        for chunk in iter(lambda: f.read(2**20), b""):
            hash.update(chunk)
        return hash.hexdigest()


def gen_hash() -> None:
    hexdigest = encrypt(args.file, args.algorithm)
    print(f"{args.algorithm}: {hexdigest}")


if __name__ == "__main__":
    gen_hash()
    # for algorithm in ('md5', 'sha1', 'sha256'):
    #     hexdigest = encrypt(r"C:\Users\\Desktop\git_update\git_update.bat", algorithm)
    #     print(f'{algorithm}: {hexdigest}')
