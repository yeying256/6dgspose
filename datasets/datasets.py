import yaml
import argparse
import logging
from pathlib import Path
import os,sys

# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

from datasets.onepose_dataset import onepose_dataset
from datasets.linemod_dataset import linemod_dataset



class datasets:
    def __init__(self,config) -> None:
        dataname = config['data']['dataname']
        if dataname == "onepose":
            self.dataset = onepose_dataset(config)
        elif dataname == "linemod":
            self.dataset = linemod_dataset(config)
        pass
    

    def choose_data(self,config):
        pass