"""
@File    :   logger.py
@Time    :   2024/06/18 21:27:12
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from typing import Union, Optional, Dict
import os
import logging
from logging.handlers import TimedRotatingFileHandler

import colorlog


__all__ = ["Logger"]


streamFormatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-9s%(asctime)s%(reset)s pid:%(process)d %(name)s line:%(lineno)s %(blue)s%(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

fileFormatter = logging.Formatter(
    fmt="%(levelname)s %(asctime)s pid:%(process)d %(name)s line:%(lineno)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)


class Logger:

    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    def __init__(
        self,
        name: str,
        level: int = logging.DEBUG,
        log_filePath: str = "",
        is_saveLog: bool = False,
        when: str = "D",
        interval: int = 1,
        backupCount: int = 0,
    ) -> None:
        """
        Initialize a logger.

        Args:
            name (str): 日志名称.
            level (int): 日志级别，默认为 logging.DEBUG.
            log_filePath (str): 日志保存路径，如果为 None，则默认保存到工作目录下的 'logs' 文件夹中. Defaults to ''.
            is_saveLog (bool): 是否保存日志文件，如果为 True，则日志文件将保存到 log_filename 指定的路径. Defaults to False.
            when (str): 日志切割的时间单位，比如 'S'（秒）、'M'（分）、'H'（小时）、'D'（天）等. Defaults to "H".
            interval (int): 日志文件切割的时间间隔，例如当 when='H' 且 interval=1 时，表示每隔一个小时进行一次切割，并生成一个新的日志文件. Defaults to 1.
            backupCount (int): 定义默认保留旧日志文件的个数（如果超过这个数量，则会自动删除最早的日志文件），默认值为 0，表示不自动删除旧日志文件. Defaults to 0.

        Returns:
            None
        """
        self.log_filePath: Optional[str] = None
        self.curren_dir = os.getcwd()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if is_saveLog:
            os.makedirs(os.path.join(self.curren_dir, "logs"), exist_ok=True)
            if log_filePath == "":
                self.log_filePath = os.path.join(self.curren_dir, "logs", f"{name}.log")
            else:
                self.log_filePath = log_filePath
            fh = TimedRotatingFileHandler(
                self.log_filePath,
                when=when,
                interval=interval,
                backupCount=backupCount,
                encoding="utf-8",
            )

            # fh = logging.FileHandler(self.log_filePath,encoding="utf-8",filemode='w')
            fh.setLevel(level)
            fh.setFormatter(fileFormatter)
            self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(streamFormatter)
        self.logger.addHandler(ch)

    def __call__(self) -> logging.Logger:
        return self.logger

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)


if __name__ == "__main__":
    logger = Logger("test", logging.DEBUG, is_saveLog=False)
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
