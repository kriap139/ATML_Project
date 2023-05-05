import os
import json
import shutil
from typing import Sequence
import cv2 as cv
import glob

def clean_dir(d: str):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

def dirUp(path: str, levels: int = 1):
    path = path.strip()
    if path.endswith(os.path.sep):
        path = path[:-1]

    for _ in range(levels):
        path = os.path.dirname(path)
    return path

def dir_name(d: str):
    if d.endswith(os.path.sep):
        d = d[:-1]
    return os.path.split(d)[1]

def copy_coco_images(coco: dict, image_dir: str, dest_dir: str, overwrite=False):
    for img in coco["images"]:
        fn = img["file_name"]
        src = os.path.join(image_dir, fn)
        dest = os.path.join(dest_dir, fn)

        if os.path.exists(dest) and not overwrite:
            pass
        else:
            shutil.copy(src, dest)

def create_coco_dataset(coco: dict, dataset_dir: str, image_dir: str, prettify=True, overwrite=False):
    if os.path.exists(dataset_dir) and overwrite:
        clean_dir(dataset_dir)
    elif os.path.exists(dataset_dir):
        return
    else:
        os.makedirs(dataset_dir)
    
    with open(f"{dataset_dir}/ann.json", mode='w') as f:
        json.dump(coco, f, indent=3 if prettify else None)
    copy_coco_images(coco, image_dir, dataset_dir, overwrite=True)

def get_dataset_name(dataset_dir: str):
    head, tail = os.path.split(dataset_dir)
    if not tail:
        head, tail = os.path.split(head)
    if not tail:
        raise RuntimeError(f"Failed to find name of dataset in directory: {dataset_dir}")
    return tail


def video_to_sequence(video_path: str, dest_dir: str, img_ext: str = '.jpg'):
    os.makedirs(dest_dir, exist_ok=True)
    cap = cv.VideoCapture(video_path)
    _, frame = cap.read()

    i = 1
    fp = os.path.join(dest_dir, i, img_ext)
    
    if os.path.exists(fp):
        for fp in glob.glob(os.path.join(dest_dir, '*')):
            fn, ext = os.path.splitext(os.path.basename(fp))
            if int(fn) > i:
                i = int(fn)

    while frame is not None:
        fp = os.path.join(dest_dir, i, img_ext)
        cv.imwrite(fp, frame)
        i += 1
        _, frame = cap.read()
    cap.release()
    

    
