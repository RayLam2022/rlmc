"""
@File    :   main.py
@Time    :   2024/06/28 22:57:09
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import warnings
import os
import os.path as osp
from glob import glob
from pprint import pprint
import random

import numpy as np
import torch


from rlmc import cfg
from rlmc import reg
from rlmc.data import datasets
from rlmc.model import predictors, models
from rlmc.trainer import Trainer, Predict
from rlmc.visualization import *


SuDict = reg["SuDict"]
Logger = reg["Logger"]

file_processor = reg["file_processor"]

warnings.filterwarnings("ignore")
logger = Logger(__name__, level=Logger.DEBUG)

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    mode = "train"
    predict_dir = r"./samples"

    # configs
    CONFIG_PATH = "/root/rlmc/rlmc/configs/trainval/segmantic_segmentation.yaml"

    args = SuDict(file_processor(CONFIG_PATH).read())
    print(args)
    train_data_dir = args.dataset.train_dir
    val_data_dir = args.dataset.val_dir

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # # dataset
    train_dataset = datasets["seg"](args, train_data_dir)
    val_dataset = datasets["seg"](args, val_data_dir)

    # model
    premodel = models["seg"](args)
    head = predictors["seg"](args)

    if args.model.is_add_predictor:
        model = torch.nn.Sequential(premodel, head)
    else:
        model = premodel

    if args.model.show_model:
        print("####################################################################")
        print("model:")
        pprint(model)

    model.to(device)

    # train
    if mode == "train":
        trainer = Trainer(args, model, train_dataset, val_dataset, device)
        trainer.train()
    elif mode == "eval":
        trainer = Trainer(args, model, train_dataset, val_dataset, device)
        trainer.evaluate()
    elif mode == "predict":
        files = glob(osp.join(predict_dir, "*.jpg"))
        predict = Predict(args, model, device)
        for file in files:
            predict.predict(file, is_save=True, is_show=True)
