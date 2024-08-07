import argparse
import os
import os.path as osp
from glob import glob
import shutil


parser = argparse.ArgumentParser("delete_jupyter_temp_dirs")
parser.add_argument(
    "-r", "--root", required=True, help="scan the temp dirs in the path"
)
parser.add_argument("-o", "--op", default="del", help="operation:show or del")
parser.add_argument("-n", "--name", default=".*ipynb_checkpoints", help="name key words")
args = parser.parse_args()


def del_jupyter_temp() -> None:
    op = args.op
    iters = glob(osp.join(args.root, "**", args.name), recursive=True)
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
    del_jupyter_temp()
