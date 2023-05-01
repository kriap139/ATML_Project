import os
import json
import shutil
from typing import Sequence
import cv2 as cv
import glob

def dirUp(path: str, levels: int):
    for _ in range(levels):
        path = os.path.dirname(path)
    return path

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
        shutil.rmtree(dataset_dir)
    elif os.path.exists(dataset_dir):
        return
    os.makedirs(dataset_dir)

    with open(f"{dataset_dir}/ann.json", mode='w') as f:
        json.dump(coco, f, indent=3 if prettify else None)
    copy_coco_images(coco, image_dir, dataset_dir, overwrite=True)

# def create_coco_train_val_datasets(coco_train: dict, coco_val: dict, dataset_dir: str, image_dir: str, prettify=True, overwrite=False):
#     train_dir = f"{dataset_dir}/train"
#     val_dir = f"{dataset_dir}/val"

#     if os.path.exists(train_dir):
#         shutil.rmtree(train_dir)
#     if os.path.exists(val_dir):
#         shutil.rmtree(val_dir)

#     os.makedirs(train_dir)
#     os.makedirs(val_dir)

#     with open(f"{train_dir}/ann.json", mode='w') as f:
#         json.dump(coco_train, f, indent=3 if prettify else None)
    
#     with open(f"{val_dir}/ann.json", mode='w') as f:
#         json.dump(coco_val, f, indent=3 if prettify else None)
    
#     copy_coco_images(coco_train, image_dir, train_dir, overwrite=overwrite)
#     copy_coco_images(coco_val, image_dir, val_dir, overwrite=overwrite)


def video_to_sequence(video_path: str, dest_dir: str, img_ext: str = '.jpg'):
    os.makedirs(dest_dir, exist_ok=True)
    cap = cv.VideoCapture(video_path)
    _, frame = cap.read()

    i = 1

    if os.path.exists(f"{dest_dir}/{i}{img_ext}"):
        for fp in glob.glob(os.path.join(dest_dir, '*')):
            fn, ext = os.path.splitext(os.path.basename(fp))
            if int(fn) > i:
                i = int(fn)

    while frame is not None:
        fp = f"{dest_dir}/{i}{img_ext}"
        cv.imwrite(fp, frame)
        i += 1
        _, frame = cap.read()
    cap.release()


if __name__ == "__main__":
    PROC_DIR = dirUp(__file__, 3)
    video_dir = f"{PROC_DIR}/data/videos"
    video =  "MOV_0376.mp4" # "MOV_0375.mp4"
    
    src = f"{video_dir}/{video}"
    dest = f"{PROC_DIR}/data/dataset/test"
    video_to_sequence(src, dest)
    

    
