# coding: utf-8
'''
Copyright (c) University of Strasbourg. All Rights Reserved.
'''
import os
import sys
import torch
import argparse
import _init_paths
import pprint

from core.inference import eval_on_lowres_mvor
from config import cfg
from utils.logger import setup_logger
from models.depthpose_x8 import get_model
from dataset.mvor import MVORDatasetTest

def parse_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser(description="DepthPose Evaluation Interface")

    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--use-cpu", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    if args.use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        pin_memory = False
    else:
        device = torch.device("cuda")
        pin_memory = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    logger = setup_logger(cfg, args.config_file, "test")
    cfg.freeze()
    test_dataset = MVORDatasetTest(
        ann_file=cfg.DATASET.TEST.ANNO_FILE, root=cfg.DATASET.TEST.ROOT_DIR
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        shuffle=cfg.DATASET.TEST.SHUFFLE,
        num_workers=cfg.DATASET.TEST.WORKERS,
        pin_memory=pin_memory,
    )
    logger.info(pprint.pformat(cfg))
    logger.info(pprint.pformat(args))

    model = get_model()
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    model.to(device)

    eval_on_lowres_mvor(cfg=cfg, test_loader=test_loader, model=model, logger=logger, device=device)


if __name__ == "__main__":
    main(parse_args())
