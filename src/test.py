from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.evaluation import DumpDetResults
import mmcv
import sys
import glob
import os
from Util import dirUp
import shutil
import json
import argparse
from enum import Enum

class Test(Enum):
    train = 1
    val = 2
    test = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return Test[s]
        except KeyError:
            raise ValueError()

def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('model', help='model file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--test-dataset', help="Dataset to test", choices=list(Test), default=Test.test)
    args = parser.parse_args()
    return args


def test_model(config_fp: str, model_fp: str, test: Test, work_dir: str):
    cfg = Config.fromfile(config_fp)
    
    out_dir = os.path.join(work_dir, str(test))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    cfg.load_from = model_fp
    cfg.work_dir = out_dir
    cfg.default_hooks['visualization'] = dict(type='DetVisualizationHook', draw=True, test_out_dir=out_dir)
    switch = True

    if test == Test.train:
        ann_file = cfg.train_dataloader.get("dataset").get("ann_file")
        data_prefix = cfg.train_dataloader.get("dataset").get("data_prefix")
    elif test == Test.val:
        ann_file = cfg.val_dataloader.get("dataset").get("ann_file")
        data_prefix = cfg.val_dataloader.get("dataset").get("data_prefix")
    else:
        switch = False
    
    if switch:
        cfg.test_dataloader.get("dataset")["ann_file"] = ann_file
        cfg.test_dataloader.get("dataset")["data_prefix"] = data_prefix
        print(cfg.test_dataloader)
        exit(-1)


    runner = Runner.from_cfg(cfg)
    runner.test_evaluator.metrics.append(
        DumpDetResults(out_file_path=os.path.join(out_dir, "results.pkl"))
    )
    runner.test()

if __name__ == "__main__":
    args = parse_args()

    if args.work_dir is None:
        args.work_dir = os.path.join(
            os.path.split(args.model)[0],
            "results")

    test_model(args.config, args.model, args.test_dataset, args.work_dir)
