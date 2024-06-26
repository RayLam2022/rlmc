"""
@File    :   gittool.py
@Time    :   2024/06/26 22:08:37
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from typing import List, Dict, Union
import os
import os.path as osp

import git


class Git:
    def __init__(self, local_repo_path=None):
        self.local_repo_path = local_repo_path
        self.local_repo = None
        self.remote_url = None
        self.remote_branchs = None
        if local_repo_path != None:
            if osp.isdir(osp.join(self.local_repo_path, ".git")):
                self.open_local_repo()

    def init_local_repo(self) -> None:  # 待
        self.local_repo = git.Repo.init(self.local_repo_path)
        origin = self.local_repo.create_remote("origin", self.local_repo.remotes.origin.url)
        origin.fetch()
        self.local_repo.create_head("main", origin.refs.main)
        self.local_repo.heads.master.set_tracking_branch(origin.refs.master)
        self.local_repo.heads.master.checkout()

    def open_local_repo(self) -> None:
        self.local_repo = git.Repo(self.local_repo_path)
        self.get_remote_info()

    def get_config(self) -> List:
        return self.local_repo.git.config("--list").split("\n")

    def set_config(self, key="https.proxy", value="https://127.0.0.1:7890"):
        self.local_repo.git.config("--global", key, value)

    def get_repo_status(self) -> str:
        return self.local_repo.git.status()

    def get_file_status(self, file_path) -> str:
        return self.local_repo.git.status(file_path)

    def get_repo_history(self) -> str:
        return self.local_repo.git.log()

    def get_file_history(self, file_path) -> str:
        return self.local_repo.git.log("--follow", "--", file_path)

    def get_commits_history(self):
        commit_history = list(self.local_repo.iter_commits())
        return commit_history

    def get_local_repo_path(self) -> str:
        return self.local_repo.working_dir

    def get_remote_info(self) -> git.remote.Remote:  
        self.remote = self.local_repo.remote()  # origin
        self.remote_url = self.remote.url
        self.remote_branchs = self.remote.refs
        return self.remote

    def get_current_branch(self) -> str:
        return self.local_repo.active_branch # master,main...

    def create_branch(self, branch_name) -> None:  
        self.local_repo.create_head(branch_name)

    def delete_branch(self, branch_name) -> None: 
        self.local_repo.delete_head(branch_name)

    def switch_branch(self, branch_name, commit_msg="update some modules") -> List: 
        self.local_repo.git.add(all=True)
        self.local_repo.git.commit("-m", commit_msg)
        self.local_repo.git.checkout(branch_name)

    def get_all_branchs(self) -> List:
        return self.local_repo.git.branch("-a").split("\n")

    def get_all_tags(self) -> List:
        """获取标签列表"""
        return self.local_repo.git.tag().split("\n")

    def get_latest_author_info(self) -> str:
        return self.local_repo.head.commit.author

    def get_diff_between_commits(self, commit_sha_1, commit_sha_2) -> List[git.diff.Diff]: 
        """获取两个提交之间的差异   git.diff.Diff可print"""   
        commit_1 = self.local_repo.commit(commit_sha_1)
        commit_2 = self.local_repo.commit(commit_sha_2)
        return commit_1.diff(commit_2)

    def get_file_content_in_commit(self, file_path, commit_sha) -> str:  
        """获取某个文件在某个提交中的内容 要用相对路径例如rltests/test.py"""
        commit = self.local_repo.commit(commit_sha)
        return commit.tree[file_path].data_stream.read().decode("utf-8")

    def cancel_uncommit_changes(self):  #
        """撤销未提交的更改"""
        self.local_repo.git.reset("--hard", "HEAD")

    def check_uncommit_changes(self) -> bool:   
        """检查是否有未提交的更改"""
        return self.local_repo.is_dirty()

    def roll_back_to_commit(self, commit_sha):  #
        """回滚到某个提交"""
        commit_to_roll_back = self.local_repo.commit(commit_sha)
        self.local_repo.git.reset("--hard", commit_to_roll_back)

    def git_push_repo(self, commit_msg="update some modules"): #
        self.local_repo.git.add(all=True)
        self.local_repo.git.commit("-m", commit_msg)
        self.local_repo.remotes.origin.push()

    def git_pull_repo(self, commit_msg="update some modules"):  #
        self.local_repo.git.add(all=True)
        self.local_repo.git.commit("-m", commit_msg)
        self.local_repo.remotes.origin.pull()

    def git_clone(self, remote_repo_url):
        git.Repo.clone_from(remote_repo_url, self.local_repo_path)
        self.open_local_repo()
        self.get_remote_info()

    def git_push_one_file(self, file_path, commit_msg="update some modules"):  
        self.local_repo.git.add(file_path)
        self.local_repo.git.commit("-m", commit_msg)
        self.local_repo.remotes.origin.push()


if __name__ == "__main__":
    local_repo_path = r"D:\work\rlmc"
    remote_repo_url = "https://github.com/RayLam2022/rlmc.git"

    gt = Git(local_repo_path)
    #gt.git_clone(remote_repo_url)
    #gt.create_branch('test_branch')
    print(gt.get_all_branchs())
    #gt.switch_branch('master')
    #x=gt.get_diff_between_commits('2a68b8e5f9bcc59e211163d5f22ab0a3fe6bbddc','2a0dc428868e7fbd3a0128876b8ef221a4f4ff23')[1]
    #x=gt.get_file_content_in_commit(r'rltests/test.py','2a0dc428868e7fbd3a0128876b8ef221a4f4ff23')
    #gt.git_pull_repo()
    print(gt.get_repo_status())
    gt.git_push_repo('create g_tools')

    #print(x)

