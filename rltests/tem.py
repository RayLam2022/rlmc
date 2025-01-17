import asyncio
import time
# a blocking io-bound task
def blocking_task():
    # report a message
    print('Task starting')
    # block for a while
    time.sleep(2)
    # report a message
    print('Task done')
    return 'done'
# main coroutine
async def main():
    # report a message
    print('Main running the blocking task')
    # create a coroutine for  the blocking task
    coro = asyncio.to_thread(blocking_task)
    print('coro',type(coro))
    # schedule the task
    task = asyncio.create_task(coro)
    print('taskk',type(task))
    # report a message
    print('Main doing other things')
    # allow the scheduled task to start
    await asyncio.sleep(0)
    # await the task
    return await task
# run the asyncio program
asyncio.run(main())
print(type(asyncio.run(main())))