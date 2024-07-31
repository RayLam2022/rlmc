import argparse
import shutil
import stat
import os
import os.path as osp
from glob import glob
import platform
import subprocess


parser = argparse.ArgumentParser("clear cache")
parser.add_argument(
    "-e", "--expand_cache_path", default=[], nargs="+",help="clear default path, and expand cache path "
)
args = parser.parse_args()

pip_cache=osp.join(osp.expanduser('~'), '.cache/pip')
conda_cache=osp.join(os.popen('conda info --root').read().strip(), 'pkgs')
trash= osp.join(osp.expanduser('~'), '.local/share/Trash')

if platform.system() == "Windows":
    args.roots = [osp.join(osp.expanduser('~'), 'AppData/Local/pip/cache'), conda_cache ] + args.expand_cache_path
elif platform.system() == "Linux":
    args.roots = [osp.join(osp.expanduser('~'), '.cache/pip'), conda_cache, trash] + args.expand_cache_path


def remove_readonly(func, path, _):  # 错误回调函数，改变只读属性位，重新删除
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_cache():
    for root in args.roots:
        if osp.exists(root):
            total_size= 0
            for f in glob(osp.join(root, "**"),recursive=True):
                if osp.isfile(f):
                    total_size+=osp.getsize(f)
            print(f'{root}: {total_size/1024/1024:.2f} MB')

    command=input("clear cache? (y/n):")
    if command=="y":
        os.system('pip cache purge')
        os.system('conda clean --all')
        for root in args.roots:
            if osp.exists(root):
                try:
                    shutil.rmtree(root, onerror=remove_readonly)  # 如报错，可能要管理员权限运行此程序
                    os.mkdir(root)
                except:
                    for f in glob(osp.join(root, "**"),recursive=True):
                        if osp.isfile(f):
                            try:
                                os.remove(f)
                            except PermissionError:
                                print(f'{f} remove failed for PermissionError')

        subprocess.run("pip uninstall rlmc -y", shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)



if __name__ == "__main__":
    clear_cache()

