#!/bin/bash

pip3 install torch torchvision

pip3 install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip3 install -v -e .

pip3 install -r requirements.txt
pip install future tensorboard