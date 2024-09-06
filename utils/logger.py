# -*- coding: utf-8 -*-

import logging
from logging import handlers
from colorlog import ColoredFormatter
import os
from os import path, mkdir


class LoggerConfig:
    """
    configure logs for output in terminal and save to file
        :param console_level: logging.DEBUG
        :param file_level: logging.ERROR
        :param path_logger: log/log.log
    """

    def __init__(
        self,
        console_level: int = logging.DEBUG,
        file_level: int = logging.ERROR,
        path_logger: str = "log/log.log",
    ):
        self.console_level = console_level
        self.file_level = file_level
        self.path_logger = path_logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        self.__configure_console()
        # self.__configure_file()

    def __configure_file(self):
        """Log configuration to save to file"""
        self.__check_path()
        file_handler = handlers.RotatingFileHandler(
            filename=self.path_logger,
            mode="a",
            maxBytes=20000000,  # 20MB
            backupCount=4,
            encoding="utf-8",
        )
        formatter = ColoredFormatter("%(levelname)s | %(asctime)s | %(pathname)s:%(lineno)d | %(message)s ")
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def __configure_console(self):
        """log configuration for console, this format creates jump link to log location"""

        # formatter = logging.Formatter(
        #    "%(levelname)s: %(asctime)s: %(pathname)s:%(lineno)d  %(message)s"
        # )
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(asctime)s%(reset)s | %(log_color)s%(pathname)s:%(log_color)s%(lineno)d%(reset)s | %(log_color)s%(message)s%(reset)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.addFilter(PackagePathFilter())
        stream_handler.setLevel(self.console_level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def __check_path(self):
        """
        check if the path exists if not create
        """
        dir_logger = path.dirname(self.path_logger)
        if not path.isdir(dir_logger):
            mkdir(dir_logger)


class PackagePathFilter(logging.Filter):
    """_summary_

    Filter to format relative file path
    the path is compatible with linux and windows

    Args:
        logging (_class_): Filter
    """

    def filter(self, record):
        """_summary_

        Args:
            record (str): logger

        Returns:
            bool: status
        """
        bar_path = os.path.join(" ", " ").strip()  # bar_path is / or \
        record.pathname = record.pathname.replace(os.getcwd() + bar_path, "")
        return True
