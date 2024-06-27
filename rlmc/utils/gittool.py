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
<<<<<<< HEAD
    """
    Not used, assist to remember the git command.
    """

    def init_local_repo(self, remote_url: str) -> None:
        """
        Initialize a local repository.
        Args:
            remote_url (str): https://github.com/xxxx/xxxx.git
        """
        self.local_repo = git.Repo.init(self.local_repo_path)
        origin = self.local_repo.create_remote("origin", remote_url)
        origin.fetch()
        self.get_remote_info()
=======
    def __init__(self, local_repo_path=None):
        self.local_repo_path = local_repo_path
        self.local_repo = None
        self.remote_url = None
        self.remote_branchs = None
        if local_repo_path != None:
            if osp.isdir(osp.join(self.local_repo_path, ".git")):
                self.open_local_repo()

    def init_local_repo(self) -> None: 
        self.local_repo = git.Repo.init(self.local_repo_path)
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9

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

<<<<<<< HEAD
    def git_stash(self, type="save") -> None:
        if type == "save":
            self.local_repo.git.stash()
        elif type =="pop":
            self.local_repo.git.stash("pop")
        elif type == "view":
            self.local_repo.git.stash("list")


    def git_stash_pop(self) -> None:
        self.local_repo.git.stash("pop")

=======
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
    def get_repo_history(self) -> str:
        return self.local_repo.git.log()

    def get_file_history(self, file_path) -> str:
        return self.local_repo.git.log("--follow", "--", file_path)

    def get_commits_history(self):
        commit_history = list(self.local_repo.iter_commits())
        return commit_history

    def get_local_repo_path(self) -> str:
        return self.local_repo.working_dir

<<<<<<< HEAD
    def get_remote_info(self) -> git.remote.Remote:
=======
    def get_remote_info(self) -> git.remote.Remote:  
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        self.remote = self.local_repo.remote()  # origin
        self.remote_url = self.remote.url
        self.remote_branchs = self.remote.refs
        return self.remote

    def get_current_branch(self) -> str:
<<<<<<< HEAD
        return self.local_repo.active_branch  # master,main...

    def create_branch(self, branch_name) -> None:
        self.local_repo.create_head(branch_name)

    def delete_branch(self, branch_name) -> None:
        self.local_repo.delete_head(branch_name)

    def switch_branch(self, branch_name) -> None:
=======
        return self.local_repo.active_branch # master,main...

    def create_branch(self, branch_name) -> None:  
        self.local_repo.create_head(branch_name)

    def delete_branch(self, branch_name) -> None: 
        self.local_repo.delete_head(branch_name)

    def switch_branch(self, branch_name) -> List: 
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        self.local_repo.git.checkout(branch_name)

    def get_all_branchs(self) -> List:
        return self.local_repo.git.branch("-a").split("\n")

    def get_all_tags(self) -> List:
        """获取标签列表"""
        return self.local_repo.git.tag().split("\n")

    def get_latest_author_info(self) -> str:
        return self.local_repo.head.commit.author

<<<<<<< HEAD
    def get_diff_between_commits(
        self, commit_sha_1, commit_sha_2
    ) -> List[git.diff.Diff]:
        """获取两个提交之间的差异   git.diff.Diff可print"""
=======
    def get_diff_between_commits(self, commit_sha_1, commit_sha_2) -> List[git.diff.Diff]: 
        """获取两个提交之间的差异   git.diff.Diff可print"""   
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        commit_1 = self.local_repo.commit(commit_sha_1)
        commit_2 = self.local_repo.commit(commit_sha_2)
        return commit_1.diff(commit_2)

<<<<<<< HEAD
    def get_file_content_in_commit(self, file_path, commit_sha) -> str:
=======
    def get_file_content_in_commit(self, file_path, commit_sha) -> str:  
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        """获取某个文件在某个提交中的内容 要用相对路径例如rltests/test.py"""
        commit = self.local_repo.commit(commit_sha)
        return commit.tree[file_path].data_stream.read().decode("utf-8")

<<<<<<< HEAD
    def get_untracked_files(self) -> List[str]:
        """获取未跟踪的文件列表"""
        return self.local_repo.untracked_files

=======
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
    def cancel_uncommit_changes(self):  #
        """撤销未提交的更改"""
        self.local_repo.git.reset("--hard", "HEAD")

<<<<<<< HEAD
    def check_uncommit_changes(self) -> bool:
=======
    def check_uncommit_changes(self) -> bool:   
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        """检查是否有未提交的更改"""
        return self.local_repo.is_dirty()

    def roll_back_to_commit(self, commit_sha):  #
        """回滚到某个提交"""
        commit_to_roll_back = self.local_repo.commit(commit_sha)
        self.local_repo.git.reset("--hard", commit_to_roll_back)

<<<<<<< HEAD
    def git_add(self, file="all"):
        if file == "all":
            self.local_repo.git.add(all=True)
        else:
            self.local_repo.git.add(file)

    def git_commit(self, commit_msg="update some modules"):
        self.local_repo.git.commit("-m", commit_msg)

    def git_merge(self, branch_name):
        self.local_repo.git.merge(branch_name)

    def git_push_repo(self, commit_msg="update some modules"):  #
        self.git_add()
        self.git_commit(commit_msg)
=======
    def git_push_repo(self, commit_msg="update some modules"):  #
        self.local_repo.git.add(all=True)
        self.local_repo.git.commit("-m", commit_msg)
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        self.local_repo.remotes.origin.push()

    def git_pull_repo(self, commit_msg="update some modules"):  #
        self.local_repo.git.add(all=True)
        self.local_repo.git.commit("-m", commit_msg)
        self.local_repo.remotes.origin.pull()

    def git_clone(self, remote_repo_url):
        git.Repo.clone_from(remote_repo_url, self.local_repo_path)
        self.open_local_repo()
        self.get_remote_info()

<<<<<<< HEAD
    def git_push_one_file(self, file_path, commit_msg="update some modules"):
=======
    def git_push_one_file(self, file_path, commit_msg="update some modules"):  #
>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
        self.local_repo.git.add(file_path)
        self.local_repo.git.commit("-m", commit_msg)
        self.local_repo.remotes.origin.push()


if __name__ == "__main__":
    local_repo_path = r"C:\Users\RayLam\Desktop\test_git"
    remote_repo_url = "https://github.com/RayLam2022/rlmc.git"

    gt = Git(local_repo_path)
<<<<<<< HEAD
    # gt.git_clone(remote_repo_url)
    # gt.git_pull_repo()
    # gt.create_branch('test_branch1')
    print(gt.get_all_branchs())
    gt.switch_branch("test_branch")
    # x=gt.get_diff_between_commits('2a68b8e5f9bcc59e211163d5f22ab0a3fe6bbddc','2a0dc428868e7fbd3a0128876b8ef221a4f4ff23')[1]
    # x=gt.get_file_content_in_commit(r'rltests/test.py','2a0dc428868e7fbd3a0128876b8ef221a4f4ff23')
    #
    # gt.git_pull_repo()
    print(gt.get_repo_status())
    print(gt.get_untracked_files())
    gt.delete_branch("remotes/origin/main")

    gt.git_push_repo("create g_tools")
    # gt.git_push_one_file(r'D:\work\rlmc\rlmc\utils\gittool.py','modify g_tools')

    # print(x)
=======
    #gt.git_clone(remote_repo_url)
    #gt.create_branch('test_branch')
    print(gt.get_all_branchs())
    gt.switch_branch('test_branch')
    #x=gt.get_diff_between_commits('2a68b8e5f9bcc59e211163d5f22ab0a3fe6bbddc','2a0dc428868e7fbd3a0128876b8ef221a4f4ff23')[1]
    #x=gt.get_file_content_in_commit(r'rltests/test.py','2a0dc428868e7fbd3a0128876b8ef221a4f4ff23')
    #gt.git_pull_repo()
    print(gt.get_repo_status())
    gt.git_push_one_file(r'D:\work\rlmc\rlmc\utils\gittool.py')

    #print(x)

>>>>>>> 1aa98bc659dcdb2937f50f5dd95f5624eaa930c9
