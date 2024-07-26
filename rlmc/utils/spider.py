"""
@File    :   spider.py
@Time    :   2024/07/24 22:45:48
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import asyncio
import random
import time
import contextvars
import functools
import os
import os.path as osp
import re

import requests
import aiohttp
import aiofiles
from lxml import etree


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

__all__ = ["Spider"]
logger = Logger(__name__, level=Logger.DEBUG)


class Spider:
    """how to use"""

    _default = {
        "headers": [
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"
            }
        ],
        "verify": False,
        "proxy": [],  # ['58.246.58.150:9002', '117.74.65.215:9080', '88.132.253.187:80']
        "timeout": 1000,
        "encoding": "gbk",  # utf-8
    }

    def __init__(
        self,
        url_list,
        producer_semaphore_num,
        consumer_func,
        consumer_semaphore_num,
    ):
        self.__dict__.update(self._default)
        self.url_list = url_list
        self.producer_semaphore_num = producer_semaphore_num
        self.consumer_semaphore_num = consumer_semaphore_num
        self.consumer_func = consumer_func
        self.result = {}

    @classmethod
    def _get_contents(cls, res):
        """
        从response decode内容，编码处理
        """
        if res is None:
            decodedata = None
        elif res.status_code == 200:
            decodedata = res.content.decode(cls._default["encoding"])
            print(cls._default["encoding"])
        else:
            decodedata = None
        return decodedata

    @classmethod
    def _gen_next_page(cls, host_url, start_url, end_url, next_page_xpath):
        current_url = start_url
        while True:
            yield current_url

            if cls._default["proxy"]:
                response = requests.get(
                    current_url,
                    timeout=cls._default["timeout"],
                    headers=cls._default["headers"][0],
                    verify=cls._default["verify"],
                    proxies=cls._default["proxy"],
                )
            else:
                response = requests.get(
                    current_url,
                    timeout=cls._default["timeout"],
                    headers=cls._default["headers"][0],
                    verify=cls._default["verify"],
                )
            decodedata = cls._get_contents(response)
            data = etree.HTML(decodedata)

            #### 看情况改 ####
            # 同步，next_page   1、测试next_page_xpath，2、再测试content_xpath（可多个），3、写记录保存代码

            # content_xpath = '' #从current url内容获取需要信息
            # content=data.xpath(content_xpath)

            if current_url == end_url:
                break

            elem = data.xpath(next_page_xpath)  # 从current url内容获取下一页
            # elem转字符查看结构： etree.tostring(elem[0])
            next_url = host_url + elem[0].attrib["href"]
            print(next_url)
            #### 看情况改 ####

            current_url = next_url

    @classmethod
    def get_all_urls(cls, host_url, start_url, end_url, xpath):
        generator = cls._gen_next_page(host_url, start_url, end_url, xpath)
        return [url for url in generator]

    async def producer(self, name, queue, elem_idx, url, semaphore, session):
        async with semaphore:
            async with session.get(url) as response:
                print(response.status)
                data = await response.text(self.encoding)
                content = await response.read()
                resp = {"text": data, "content": content}
                await queue.put((elem_idx, url, resp))
                logger.info(f"{name} is produced item: {url}")

    async def consumer(self, name, queue):
        #### 看情况改 ####
        # 异步处理只要改consumer即可
        #### 看情况改 ####
        while True:
            elem_idx, url, item = await queue.get()
            res = await asyncio.create_task(
                async_to_thread(self.consumer_func, url, item)
            )

            queue.task_done()
            self.result[elem_idx] = (url, res)
            logger.info(f"{name} is working on item: {url}")

    async def run(self):
        queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(self.producer_semaphore_num)

        conn = aiohttp.TCPConnector(ssl=self.verify, limit=None)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(
            headers=self.headers[0], connector=conn, timeout=timeout
        ) as session:
            producer_tasks = []
            for idx, url in enumerate(self.url_list):
                task = asyncio.create_task(
                    self.producer(
                        f"producer-worker-{idx}", queue, idx, url, semaphore, session
                    )
                )
                producer_tasks.append(task)

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

    async def main(self):  # 重要，不套一个await会没有结果
        await self.run()


def sync_co_task(url, resp):
    print("--------------------------")
    # print(type(resp['text']))
    data = etree.HTML(resp["text"])
    elem = data.xpath("//div")
    # print(etree.tostring(data))
    # print(elem[20].text)
    content = ""
    for ele in elem:
        if ele is not None:  # 重要
            if ele.text is not None:  # 重要
                content += ele.text

    return content


if __name__ == "__main__":
    ##同步 next page下一页##
    # host_url = "https://www..com"
    # start_url = "https://www..com/2021n/12/19118.html"
    # end_url = "https://www..com/2021n/12/19118_4.html"
    # next_page_xpath = "//a[contains(text(),'下一页')]"
    # urls = Spider.get_all_urls(host_url, start_url, end_url, next_page_xpath)
    # print(urls)
    ##同步 next page下一页##

    ##异步 已知url列表##
    start = time.time()
    task_list = [
        "https://www..com/2021n/12/19118_4.html",
    ]
    # task_list=['https://www..info/266/266564/76434073.html']
    spider = Spider(
        task_list,
        producer_semaphore_num=3,
        consumer_func=sync_co_task,
        consumer_semaphore_num=3,
    )

    main = spider.main()
    asyncio.run(main)
    print(spider.result)
    print(time.time() - start)
