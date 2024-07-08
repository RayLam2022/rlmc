import sys

if "." not in sys.path:
    sys.path.append(".")

import argparse
import platform
import os
import os.path as osp
from rlmc.resource import condarc, pip

parser = argparse.ArgumentParser("manage source")
parser.add_argument(
    "-s", "--source", default='pip', help="pip, conda, apt"
)
args = parser.parse_args()

codename=os.popen('cat /etc/os-release | grep UBUNTU_CODENAME').read().strip().split("=")[1]
apt = f"deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename} main restricted\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-updates main restricted\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename} universe\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-updates universe\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename} multiverse\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-updates multiverse\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-backports main restricted universe multiverse\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-security main restricted\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-security universe\n \
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ {codename}-security multiverse"


def manage_source():
    if platform.system() == "Windows":
        pip_src = osp.join(osp.expanduser("~"), "pip/pip.ini")
    elif platform.system() == "Linux":
        pip_src = osp.join(osp.expanduser("~"), ".pip/pip.conf")

    conda_src = osp.join(osp.expanduser("~"), ".condarc")
    if args.source == "pip":
        if osp.exists(pip_src):
            print("####################### pip ori src #######################\n")
            with open(pip_src, "r") as f:
                print(f.read())

        command = input("change pip source? (y/n):")
        if command == "y":
            os.makedirs(osp.dirname(pip_src), exist_ok=True)
            with open(pip_src, "w") as f:
                f.write(pip)
    elif args.source == "conda":
        if osp.exists(conda_src):
            print("####################### conda ori src #######################\n")
            with open(conda_src, "r") as f:
                print(f.read())

        command = input("change conda source? (y/n):")
        if command == "y":
            with open(conda_src, "w") as f:
                f.write(condarc)
    
    elif args.source == "apt":
        if platform.system() == "Linux":
            apt_src='/etc/apt/sources.list'
            if osp.exists(apt_src):
                print("####################### apt ori src #######################\n")
                with open(apt_src, "r") as f:
                    print(f.read())
            command = input("change apt source? (y/n):")
            if command == "y":
                if not osp.exists('/etc/apt/sources.list.bak'):
                    os.system(f'cp {apt_src} /etc/apt/sources.list.bak')
                with open(apt_src, "w") as f:
                    f.write(apt)



if __name__ == "__main__":
    manage_source()
