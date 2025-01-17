"""
@File    :   asynctask.py
@Time    :   2024/06/21 13:08:38
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys
import pathlib


if "." not in sys.path:
    sys.path.append(".")

import asyncio
import contextvars
import functools
import time
import random
from abc import ABC, abstractmethod
from typing import Callable, Union, Dict, List, Any


from rlmc.utils.logger import Logger



python_version = f"{sys.version_info.major}.{sys.version_info.minor}"


# ---------python3.8 没有asyncio.to_thread，此处支持python3.8--------*
async def to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


try:
    import asyncio.to_thread as async_to_thread
except:
    async_to_thread = to_thread

# ---------python3.8 没有asyncio.to_thread，此处支持python3.8--------*

__all__ = ["AsyncTasks", "AsyncProducerConsumer", "AsyncProducerConsumerTriple"]

logger = Logger(__name__, level=Logger.DEBUG)


class AsyncTasks:
    """how to use
    if __name__ == "__main__":
        start = time.time()
        asynctasks = AsyncTasks(semaphore_num=2)

        task1 = asynctasks.create_task(sync_task, 2, 9)
        task2 = asynctasks.create_task(sync_task, 2, 9)
        task3 = asynctasks.create_task(sync_task, 3, 6)

        main = asynctasks.main(*[task1, task2, task3])
        res = asyncio.run(main)
        print(res)
        print(time.time() - start)
    """

    def __init__(self, semaphore_num: int) -> None:
        self.semaphore_num = semaphore_num

    async def _async_wrapper(self, sync_func: Callable, *args, **kwargs):
        coroutine = await async_to_thread(sync_func, *args, **kwargs)
        return coroutine

    async def _semaphore(self, semaphore: asyncio.Semaphore, task: asyncio.Task):
        # 控制异步并发数，with semaphore要套每个task
        async with semaphore:
            return await task

    async def create_task(self, sync_func: Callable, *args, **kwargs) -> asyncio.Task:
        task = await asyncio.create_task(
            self._async_wrapper(sync_func, *args, **kwargs)
        )
        return task

    async def main(self, *tasks) -> Any:
        # semaphore初始化要在gather前
        semaphore = asyncio.Semaphore(self.semaphore_num)
        tasks = [self._semaphore(semaphore, task) for task in tasks]
        res = await asyncio.gather(*tasks)
        return res


class AsyncProducerConsumer:
    """how to use
    if __name__ == "__main__":
        start = time.time()
        task_list = [2, 0.2, 0, 1, 1, 4, 0.7]
        asyncproductorconsumer = AsyncProducerConsumer(
            task_list,
            producer_func=sync_pro_task,
            producer_semaphore_num=7,
            consumer_func=sync_co_task,
            consumer_semaphore_num=7,
        )

        main = asyncproductorconsumer.main()
        asyncio.run(main)
        print(asyncproductorconsumer.result)
        print(time.time() - start)
    """

    def __init__(
        self,
        element_list,
        producer_func,
        producer_semaphore_num,
        consumer_func,
        consumer_semaphore_num,
    ):
        self.producer_semaphore_num = producer_semaphore_num
        self.consumer_semaphore_num = consumer_semaphore_num
        self.element_list = element_list
        self.producer_func = producer_func
        self.consumer_func = consumer_func
        self.result = {}

    async def producer(self, name, queue: asyncio.Queue, elem_idx, elem, semaphore):
        async with semaphore:
            coroutine = async_to_thread(self.producer_func, elem)
            item = await asyncio.create_task(coroutine)
            await queue.put((elem_idx, elem, item))
            logger.info(f"{name} is produced item: {item}")

    async def consumer(self, name, queue: asyncio.Queue):
        while True:
            elem_idx, elem, item = await queue.get()
            res = await asyncio.create_task(async_to_thread(self.consumer_func, item))
            queue.task_done()
            self.result[elem_idx] = (elem, res)
            logger.info(f"{name} is working on item: {item}")

    async def main(self):
        queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(self.producer_semaphore_num)

        producer_tasks = []
        for idx, elem in enumerate(self.element_list):
            task = asyncio.create_task(
                self.producer(f"producer-worker-{idx}", queue, idx, elem, semaphore)
            )
            producer_tasks.append(task)

        # total_sleep_time = 0
        # for _ in range(20):
        #     sleep_for = random.uniform(0.05, 1.0)
        #     total_sleep_time += sleep_for
        #     queue.put_nowait(sleep_for)

        tasks = []
        for i in range(self.consumer_semaphore_num):
            task = asyncio.create_task(self.consumer(f"consumer-worker-{i}", queue))
            tasks.append(task)

        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*producer_tasks, return_exceptions=True)

        # Wait until the queue is fully processed.
        started_at = time.monotonic()
        await queue.join()
        total_slept_for = time.monotonic() - started_at

        # Cancel our worker tasks.
        for task in tasks:
            task.cancel()
        return tasks


class AsyncProducerConsumerTriple:
    """how to use
    if __name__ == "__main__":
        start = time.time()
        task_list = [1, 0.2, 0, 1, 1, 1, 0.7,0.8,0.9,1,1]
        asyncproductorconsumer = AsyncProducerConsumerTriple(
            task_list,
            producer_func=sync_pro_task,
            producer_semaphore_num=14,
            middle_func=sync_co_task,
            middle_semaphore_num=10,
            consumer_func=sync_co_task,
            consumer_semaphore_num=9,
        )

        main = asyncproductorconsumer.main()
        asyncio.run(main)
        print(asyncproductorconsumer.result)
        print(time.time() - start)
    """

    def __init__(
        self,
        element_list,
        producer_func,
        producer_semaphore_num,
        middle_func,
        middle_semaphore_num,
        consumer_func,
        consumer_semaphore_num,
    ):
        self.producer_semaphore_num = producer_semaphore_num
        self.consumer_semaphore_num = consumer_semaphore_num
        self.middle_func = middle_func
        self.middle_semaphore_num = middle_semaphore_num
        self.element_list = element_list
        self.producer_func = producer_func
        self.consumer_func = consumer_func
        self.result = {}

    async def producer(self, name, queue, elem_idx, elem, semaphore):
        async with semaphore:
            coroutine = async_to_thread(self.producer_func, elem)
            item = await asyncio.create_task(coroutine)
            await queue.put((elem_idx, elem, item))
            logger.info(f"{name} is produced item: {item}")

    async def consumer(self, name, queue):
        while True:
            elem_idx, elem, item = await queue.get()
            res = await asyncio.create_task(async_to_thread(self.consumer_func, item))
            queue.task_done()
            self.result[elem_idx] = (elem, res)
            logger.info(f"{name} is working on item: {item}")

    async def middle(self, name, queue_from_producer, queue_to_consumer):
        while True:
            elem_idx, elem, item = await queue_from_producer.get()
            item = await asyncio.create_task(async_to_thread(self.middle_func, item))
            queue_from_producer.task_done()
            await queue_to_consumer.put((elem_idx, elem, item))
            logger.info(f"{name} is middle working on item: {item}")

    async def main(self):
        queue_producer = asyncio.Queue()
        queue_middle = asyncio.Queue()
        semaphore = asyncio.Semaphore(self.producer_semaphore_num)

        producer_tasks = []
        for idx, elem in enumerate(self.element_list):
            task = asyncio.create_task(
                self.producer(
                    f"producer-worker-{idx}", queue_producer, idx, elem, semaphore
                )
            )
            producer_tasks.append(task)

        tasks_middle = []
        for i in range(self.middle_semaphore_num):
            task = asyncio.create_task(
                self.middle(f"middle-worker-{i}", queue_producer, queue_middle)
            )
            tasks_middle.append(task)

        tasks = []
        for i in range(self.consumer_semaphore_num):
            task = asyncio.create_task(
                self.consumer(f"consumer-worker-{i}", queue_middle)
            )
            tasks.append(task)

        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*producer_tasks, return_exceptions=True)

        # Wait until the queue is fully processed.
        started_at = time.monotonic()
        await queue_producer.join()
        await queue_middle.join()
        total_slept_for = time.monotonic() - started_at

        # Cancel our worker tasks.
        for task in tasks:
            task.cancel()
        return tasks


# 一个原同步函数
def sync_pro_task(interval):
    time.sleep(interval)
    return interval


def sync_co_task(interval):
    time.sleep(interval)
    return interval + 1


def sync_task(interval, num):
    time.sleep(interval)
    return f"{interval}_{num}"


if __name__ == "__main__":
    # start = time.time()
    # asynctasks = AsyncTasks(semaphore_num=2)

    # task1 = asynctasks.create_task(sync_task, 2, 9)
    # task2 = asynctasks.create_task(sync_task, 2, 9)
    # task3 = asynctasks.create_task(sync_task, 3, 6)

    # main = asynctasks.main(*[task1, task2, task3])
    # res = asyncio.run(main)
    # print(res)
    # print(time.time() - start)

    # start = time.time()
    # task_list = [2, 0.2, 0, 1, 1, 4, 0.7]
    # asyncproductorconsumer = AsyncProducerConsumer(
    #     task_list,
    #     producer_func=sync_pro_task,
    #     producer_semaphore_num=7,
    #     consumer_func=sync_co_task,
    #     consumer_semaphore_num=7,
    # )

    # main = asyncproductorconsumer.main()
    # asyncio.run(main)
    # print(asyncproductorconsumer.result)
    # print(time.time() - start)

    start = time.time()
    task_list = [1, 0.2, 0, 1, 1, 1, 0.7, 0.8, 0.9, 1, 1]
    asyncproductorconsumer = AsyncProducerConsumerTriple(
        task_list,
        producer_func=sync_pro_task,
        producer_semaphore_num=14,
        middle_func=sync_co_task,
        middle_semaphore_num=10,
        consumer_func=sync_co_task,
        consumer_semaphore_num=9,
    )

    main = asyncproductorconsumer.main()
    asyncio.run(main)
    print(asyncproductorconsumer.result)
    print(time.time() - start)
