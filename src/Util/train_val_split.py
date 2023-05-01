import copy
from typing import Tuple, Dict, Sequence, Union
import os
import glob
import random

def load_val_paths(val_dir: str):
    paths = {}
    for img_fp in glob.glob(os.path.join(val_dir, '*')):
        paths[os.path.basename(img_fp)] = img_fp
    return paths

def random_coco_val_images(coco: dict, n_samples: Union[float, int]) -> tuple:
    images = coco["images"]
    if type(n_samples) == float:
        n_samples = int(len(images) * n_samples)
        
    sample = random.sample(images, k=n_samples)
    return tuple(img["file_name"] for img in sample)

def create_train_test_split(coco: dict, val_img_names: Tuple[str, ...]) -> Tuple[dict, dict]:
    anns_by_img = {img["id"]: [] for img in coco["images"]}

    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    
    coco_train = copy.deepcopy(coco)
    coco_val = copy.deepcopy(coco)

    coco_train["annotations"].clear()
    coco_val["annotations"].clear()
    coco_train["images"].clear()
    coco_val["images"].clear()

    assert len(coco_train["annotations"]) == 0
    assert len(coco_val["annotations"]) == 0
    assert len(coco_train["images"]) == 0
    assert len(coco_val["images"]) == 0


    for img in coco["images"]:
        if img['file_name'] in val_img_names:
            coco_val["images"].append(img)
            coco_val["annotations"].extend(anns_by_img[img["id"]])
        else:
            coco_train["images"].append(img)
            coco_train["annotations"].extend(anns_by_img[img["id"]])
    
    return coco_train, coco_val