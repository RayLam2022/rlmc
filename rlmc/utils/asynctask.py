"""
@File    :   asynctask.py
@Time    :   2024/06/21 13:08:38
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

sys.path.append(".")

import asyncio
import time


from rlmc.utils.logger import Logger


__all__ = ["AsyncTasks"]

logger = Logger(__name__, level=Logger.DEBUG)


class AsyncTasks:
    def __init__(self): ...

    async def async_wrapper(self, sync_func, *args, **kwargs):
        coroutine = await asyncio.to_thread(sync_func, *args, **kwargs)
        return coroutine

    async def create_task(self, sync_func, *args, **kwargs):
        task = await asyncio.create_task(self.async_wrapper(sync_func, *args, **kwargs))
        return task

    async def run(self, *tasks):
        res = await asyncio.gather(*tasks)
        return res


# 一个原同步函数
def sync_task(interval, num):
    time.sleep(interval)
    return f"task_{interval + num}"


# 一个异步函数
# async def async_task(sync_task, *args, **kwargs):
#     task = asyncio.to_thread(sync_task, *args, **kwargs)
#     res = await task
#     return res


# async def main():
#     print("main start")

#     task1 = asyncio.create_task(async_task(sync_task, 1))

#     task2 = asyncio.create_task(async_task(sync_task, 2))

#     print("main end")

#     ret1 = await task1
#     ret2 = await task2
#     print(ret1, ret2)


if __name__ == "__main__":
    start = time.time()
    # asyncio.run(main())

    asynctasks = AsyncTasks()
    task1 = asynctasks.create_task(sync_task, 1, 9)
    task2 = asynctasks.create_task(sync_task, 2, 9)
    task3 = asynctasks.create_task(sync_task, 3, 6)

    main = asynctasks.run(task1, task2, task3)
    res = asyncio.run(main)
    print(res)
    print(time.time() - start)

    start = time.time()
    print(sync_task(1, 9))
    print(sync_task(2, 9))
    print(sync_task(3, 6))
    print(time.time() - start)
