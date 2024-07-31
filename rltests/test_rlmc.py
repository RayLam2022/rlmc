import sys

if "." not in sys.path:
    sys.path.append(".")

from rlmc import cfg
from rlmc import reg

def test_yaml():
    file_processor = reg["file_processor"]
    yamlobj = file_processor("rlmc/configs/common_urls.yaml")
    yaml_content = yamlobj.data
    assert yaml_content != "", "yaml content is empty"
    print(yaml_content)

def test_cfg():
    assert cfg.source.cuda != "", "cfg content is empty"
    print(cfg.source.cuda)


if __name__ == "__main__":
    # MsDownload=reg['AutodlDownload']
    # repo_id="tzwm/StableDiffusion-others/res101.pth"
    # ms_download=MsDownload(repo_id)
    # ms_download.run()
    file_processor = reg["file_processor"]
    yamlobj = file_processor("rlmc/configs/common_urls.yaml")
    yaml_content = yamlobj.data
    print(yaml_content)
    # print(cfg['source']['cuda'])
    print(cfg.source.cuda)