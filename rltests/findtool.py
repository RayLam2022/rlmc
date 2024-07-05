import argparse
import os
import os.path as osp
import glob
import re

from tqdm import tqdm

parser = argparse.ArgumentParser("findtool")
parser.add_argument("-r", "--root", required=True, help="scan the path")
parser.add_argument("-n", "--name", default="*", help="name except extension")
parser.add_argument(
    "-e", "--exts", nargs="+", help="extension, like .py .conf if empty, fill * "
)
parser.add_argument(
    "-rc", "--recursive", action="store_true", help="glob files not recursive"
)
parser.add_argument(
    "-d", "--dir", action="store_true", help="not find the files, find dir"
)
parser.add_argument(
    "-rg",
    "--regex",
    action="store_true",
    help="is use regex to find content",
)
parser.add_argument(
    "-c", "--content", default="", help="find the content,if empty,find all files"
)
parser.add_argument("--encoding", default="utf-8")
args = parser.parse_args()


def find_string_in_file(file_path, search_string):
    with open(file_path, "r", encoding=args.encoding) as file:
        for line in file.readlines():
            if args.regex:
                if re.search(search_string, line):
                    return True
            else:
                if search_string in line:
                    return True
    return False


def find():
    files = []
    if args.exts:
        for ext in args.exts:
            if args.recursive:
                files.extend(
                    glob.glob(
                        osp.join(args.root, "**", args.name + ext), recursive=True
                    )
                )
            else:
                files.extend(
                    glob.glob(osp.join(args.root, args.name + ext), recursive=False)
                )
    else:
        if not args.dir:
            if args.recursive:
                files.extend(
                    glob.glob(
                        osp.join(args.root, "**", args.name + ".*"), recursive=True
                    )
                )
            else:
                files.extend(
                    glob.glob(osp.join(args.root, args.name + ".*"), recursive=False)
                )
        else:
            if args.recursive:
                files.extend(
                    glob.glob(
                        osp.join(args.root, "**", args.name + "/"), recursive=True
                    )
                )
            else:
                files.extend(
                    glob.glob(osp.join(args.root, args.name + "/"), recursive=False)
                )

    if args.dir:
        files = [f for f in files if osp.isdir(f)]
    else:
        files = [f for f in files if osp.isfile(f)]
    print(files)
    print(len(files))

    if args.content != "":
        collector = []
        for file in tqdm(files):
            try:
                is_found = find_string_in_file(file, args.content)
            except UnicodeDecodeError as ue:
                print(f"\nskip:UnicodeDecodeError processing file {file}: {ue}")
                is_found = False
            except Exception as e:
                print(f"\nskip:Error processing file {file}: {e}")
                is_found = False
            if is_found:
                collector.append(file)

        if collector:
            print('result:', collector)
        else:
            print("No files match")


if __name__ == "__main__":
    find()
