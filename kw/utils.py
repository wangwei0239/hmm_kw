#! /usr/bin/env python3
# coding: utf-8

import sys
import logging
import time
from datetime import datetime as dt

sys.path.insert(0, '..')
from .config import LOG_PATH


def date2timestamp(s):
    # date = "yyyy-mm-dd %H:%M:%S"
    return int(time.mktime(dt.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))


def timestamp2date(s, format_str="%Y-%m-%d %H:%M:%S"):
    return dt.fromtimestamp(int(s)).strftime(format_str)


INFO = logging.INFO
WARN = logging.WARN
DEBUG = logging.DEBUG
ERROR = logging.ERROR

COLOR = {
    INFO: '\033[92m',  # 绿
    ERROR: '\033[91m',  # 红
    WARN: '\033[93m',   # 黄
    DEBUG: '\033[43m',
    "end": "\33[0m"
}

LOG_FORMATTER = logging.Formatter('%(asctime)s [%(module)s:%(lineno)04d %(funcName)s] %(levelname)s: %(message)s')


# 打印上色
def wrapstring(string, level=INFO):
    return COLOR[level] + string + COLOR["end"]


def create_file_handler(log_name, level=logging.DEBUG, formatter=LOG_FORMATTER):
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    return file_handler


def create_stream_handler(level=logging.DEBUG, formatter=LOG_FORMATTER):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    return stream_handler


def get_logger(log_name, extra_error_file=False, is_stream_handler=True):
    logger = logging.getLogger(log_name)
    #info_file_handler = create_file_handler(os.path.join(LOG_PATH, log_name+".log"))
    #logger.addHandler(info_file_handler)
    #if extra_error_file:
    #    error_file_handler = create_file_handler(os.path.join(LOG_PATH, log_name + "_error.log"), level=logging.ERROR)
    #    logger.addHandler(error_file_handler)
    if is_stream_handler:
        logger.addHandler(create_stream_handler())
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger("keyword")
logging.getLogger("keyword").propagate = False
