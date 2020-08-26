import os
from yacs.config import CfgNode as CN

_C = CN()


_C.OUTPUT_DIR = ""
_C.LOG_DIR = ""
_C.FINAL_OUTPUT_DIR = ""
_C.FINAL_LOG_DIR = ""
_C.GPUS = (0,)
_C.PRINT_FREQ = 20


_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


_C.MODEL = CN()
_C.MODEL.NAME = ""



# DATASET related params
_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.ROOT_DIR = ""
_C.DATASET.TRAIN.ANNO_FILE = ""
_C.DATASET.TRAIN.BATCH_SIZE = 12
_C.DATASET.TRAIN.SHUFFLE = True
_C.DATASET.TRAIN.WORKERS = 4

_C.DATASET.VAL = CN()
_C.DATASET.VAL.ROOT_DIR = ""
_C.DATASET.VAL.ANNO_FILE = ""
_C.DATASET.VAL.BATCH_SIZE = 12
_C.DATASET.VAL.SHUFFLE = True
_C.DATASET.VAL.WORKERS = 4

_C.DATASET.TEST = CN()
_C.DATASET.TEST.ROOT_DIR = ""
_C.DATASET.TEST.ANNO_FILE = ""
_C.DATASET.TEST.BATCH_SIZE = 12
_C.DATASET.TEST.SHUFFLE = True
_C.DATASET.TEST.WORKERS = 4

# testing
_C.TEST = CN()
_C.TEST.FLIP_TEST = True
_C.TEST.SCALE = 10
_C.TEST.POSE_THRESH1 = 0.1
_C.TEST.POSE_THRESH2 = 0.05
_C.TEST.POSE_THRESH3 = 0.5
_C.TEST.MODEL_FILE = ""
_C.TEST.OUTPUT_JSON_FILE = ""



