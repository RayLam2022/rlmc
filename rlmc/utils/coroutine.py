"""
@File    :   coroutine.py
@Time    :   2024/06/21 09:02:18
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

sys.path.append(".")

import time
from rlmc.utils.logger import Logger


__all__ = ["Abstract_ManMachineChat"]

logger = Logger(__name__, level=Logger.DEBUG)


from abc import ABCMeta, abstractmethod


class Abstract_ManMachineChat(metaclass=ABCMeta):
    @abstractmethod
    def person(self):
        msg = ""
        while True:
            msg = yield msg
            logger.info(f"person 1:{msg}")
            time.sleep(1)

    @abstractmethod
    def computer(self):
        msg = ""
        while True:
            msg = yield msg
            logger.info(f"computer 2:{msg},i'm computer")
            time.sleep(1)

    @abstractmethod
    def runchat(self):
        task1 = self.person()
        task2 = self.computer()
        next(task1)
        next(task2)
        while True:
            msg1 = task1.send(input("person 1 input:"))
            msg2 = task2.send(msg1)
            print(msg2)


class ManMachineChatExample(Abstract_ManMachineChat):
    def person(self):
        msg = ""
        while True:
            msg = yield msg
            logger.debug(f"person 1:{msg}")
            time.sleep(1)

    def computer(self):
        msg = ""
        while True:
            msg = yield msg + 'yield1'
            logger.debug(f"computer 21:{msg},i'm computer")
            msg = yield msg + 'yield2'
            logger.debug(f"computer 22:{msg}")
            time.sleep(1)

    def runchat(self):
        task1 = self.person()
        task2 = self.computer()
        next(task1)
        next(task2)
        while True:
            msg1 = task1.send(input("person 1 input:"))
            msg2 = task2.send(msg1)
            print(msg2)


def task1():
    msg = ""
    epoch=0
    while True:
        msg = yield f'task_yield1:{epoch}' 
        print(f'task_{msg}_1')
        msg = yield f'task_yield2:{epoch}' 
        print(f'task_{msg}_2')
        print('task_n:',epoch)
        epoch+=1
        time.sleep(1)
        if epoch>6:   # 最好在task内控制轮数
            break


def how_to_understand_yield():
    t1= task1()
    next(t1)
    counter=0
    while True:
        try:
            msg1 = t1.send(input("person 1 input:"))
        except:
            break
        else:
            print('how_counter:',counter)
            print('how_return_msg:',msg1)
            counter+=1

if __name__ == "__main__":
    # chat = ManMachineChatExample()
    # chat.runchat()

    how_to_understand_yield()
