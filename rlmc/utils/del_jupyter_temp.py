import argparse
import os
import os.path as osp
from glob import glob
import shutil


parser = argparse.ArgumentParser("delete_jupyter_temp_dirs")
parser.add_argument(
    "-r", "--root", required=True, help="scan the temp dirs in the path"
)
parser.add_argument("-o", "--op", default="del", help="scan the temp dirs in the path")
parser.add_argument("-n", "--name", default=".*ipynb_checkpoints", help="key words")
args = parser.parse_args()


def del_jupyter_temp(
    root: str, name: str = ".*ipynb_checkpoints", op: str = "show"
) -> None:
    iters = glob(osp.join(root, "**", name), recursive=True)
    if op == "show":
        print(iters)
    elif op == "del":
        for i in iters:
            if osp.isfile(i):
                os.remove(i)
            else:
                shutil.rmtree(i)
            print(i)


if __name__ == "__main__":
    del_jupyter_temp(args.root, name=args.name, op=args.op)
