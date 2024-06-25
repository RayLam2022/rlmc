import sys

if '.' not in sys.path: sys.path.append(".")

from rlmc import cfg
from rlmc import reg

if __name__=='__main__':
    MsDownload=reg['AutodlDownload']
    repo_id="tzwm/StableDiffusion-others/res101.pth"
    ms_download=MsDownload(repo_id)
    ms_download.run()
