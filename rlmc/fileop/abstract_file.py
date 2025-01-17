'''
@File    :   abstract_file.py
@Time    :   2024/06/21 20:46:38
@Author  :   RayLam
@Contact :   1027196450@qq.com
'''
from abc import ABC, abstractmethod

class AbstractFile(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass


    
