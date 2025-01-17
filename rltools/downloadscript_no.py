from huggingface_hub import snapshot_download
import os
import shutil


def download_repo(repo_id, target_dir="./merged_repo",cache_dir="./"):
    try:
        repo_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
        print(f"Repository downloaded successfully: {repo_path}")

        for root, _, files in os.walk(repo_path):
            for file in files:
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(src_file, repo_path)
                dest_file = os.path.join(target_dir, relative_path)

                dest_dir = os.path.dirname(dest_file)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                shutil.copy2(src_file, dest_file)
        print(f"Repository files merged successfully into: {target_dir}")
    except Exception as e:
        print(f"Error downloading repository: {e}")

# usage example
REPO_ID = "Yongxin-Guo/trace-uni"
target_dir = "/root/autodl-tmp/trace_uni"
cache_dir='/root/autodl-tmp/cache'
download_repo(REPO_ID, target_dir=target_dir,cache_dir=cache_dir)

