#! /usr/bin/env python3
# coding: utf-8

import os


THIS_PATH = os.path.join(os.path.abspath(os.curdir))
LOG_PATH = os.path.join(THIS_PATH, "log")

MODEL_PERSISTENCE = 36000 if "MODEL_PERSISTENCE" not in os.environ else int(os.environ["MODEL_PERSISTENCE"])  # 10h
MODEL_STACK_CAPACITY = 10 if "MODEL_STACK_CAPACITY" not in os.environ else int(os.environ["MODEL_STACK_CAPACITY"])

DEFALUT_ITERATION = 10 if "DEFALUT_ITERATION" not in os.environ else int(os.environ["DEFALUT_ITERATION"])

MONGO_HOST = "172.16.101.61" if "MONGO_HOST" not in os.environ else os.environ["MONGO_HOST"]
MONGO_PORT = 27017 if "MONGO_PORT" not in os.environ else int(os.environ["MONGO_PORT"])
MONGO_DB_NAME = "correction" if "MONGO_DB_NAME" not in os.environ else os.environ["MONGO_DB_NAME"]
CELERY_BROKER = "mongodb://172.16.101.61:27017/broker" if "CELERY_BROKER" not in os.environ else os.environ["CELERY_BROKER"]
