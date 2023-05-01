import json
import sys
import cv2
import os
import glob
from Util.drawing import show_draw_contours, save_draw_contours
from typing import Union
import shutil
from pycocotools.coco import COCO
from Util.cio import dirUp
import matplotlib.pyplot as plt
import numpy as np


def coco_draw_masks_test(coco_fp: str, image_dir: str, cat_draw_order: dict, colors_by_cat: dict, output_dir: str, alpha: float = 0.9):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    coco = COCO(coco_fp)

    categories = coco.loadCats(coco.getCatIds())
    labels = [category['name'] for category in categories]
    super_labels = set([category['supercategory'] for category in categories])

    print(f"COCO labels: {', '.join(labels)}")
    print(f"COCO label categories: {', '.join(super_labels)}")

    images = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds()))
    
    for img_info in images:
        fn = img_info["file_name"]
        fp = os.path.join(image_dir, fn)
        

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=coco.getCatIds(), iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        anns.sort(key=lambda ann: cat_draw_order[ann["category_id"]])
        
        img = cv2.imread(fp)
        overlay = img.copy()

        for ann in anns:
            mask = coco.annToMask(ann)
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            color = colors_by_cat[ann["category_id"]]
            cv2.drawContours(overlay, contours, -1, color, -1)
            
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.imwrite(os.path.join(output_dir, fn), img)




def check_coco_ann_file(fp: str) -> dict:
    with open(fp, mode='r') as f:
        coco = json.load(f)
    
    task_info_fp = os.path.join(os.path.split(fp)[0], "image_task_mappings.json")
    with open(task_info_fp, mode='r') as f:
        img_to_task: dict = json.load(f)
        img_to_task = {val: key for key, val in img_to_task.items()}
    
    anns = coco["annotations"]
    images = {img['id']: img for img in coco["images"]}  
    result = {img["file_name"]: [] for img in images.values()}

    for ann in anns:
        bbox = ann["bbox"]
        segs = ann["segmentation"]

        assert len(bbox) == 4
        img_fn = images[ann['image_id']]['file_name']
        task_id = img_to_task[img_fn]

        if (len(segs) == 1) and len(segs[0]) <= 4:
            print(f"Length of segment in annotation {ann['id']} is <= 4: {segs[0]} (image={img_fn}, task={task_id})")
            result[img_fn].append(task_id)
        elif len(segs) == 0:
            raise RuntimeError(f"no segments in annotation: {ann['id']} (image={img_fn})")
        else:
            for seg in segs:
                if len(seg) <= 4:
                    print(f"Length of partial segment in annotation {ann['id']} is <= 4: {seg} (image={img_fn}, task={task_id})")
                    result[img_fn].append(task_id)
    
    result = {key: arr for key, arr in result.items() if len(arr) > 0}
    return result


def analyse_masks(masks_dir: str, task_ids: Union[str, list], out_dir: str):
    if type(task_ids) == str:
            task_ids = (task_ids, )

    for mask_fp in glob.glob(os.path.join(masks_dir, '*')):
        mask_fn = os.path.basename(mask_fp)

        # task-1904-annotation-2-by-1-tag-Valve_Lever_Brown-0.png
        parts = mask_fn.split("-")
        assert parts[0] == "task"

        if parts[1] in task_ids:
            mask_img = cv2.imread(mask_fp)

            gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            for cnt in contours:
                if len(cnt) <= 4:
                    save_draw_contours(masks_dir, mask_fn, out_dir, [cnt], (0, 0, 255))


def check_coco(coco_fp: str):
    check_fails = check_coco_ann_file(coco_fp)

    dataset = os.path.dirname(coco_fp)
    PROC_DIR = os.path.split(coco_fp)[0]    
    masks_dir = os.path.join(PROC_DIR, "masks")

    out_dir = os.path.join(PROC_DIR, "masks-analysis")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for img_fn, tasks in check_fails.items():
        analyse_masks(masks_dir, tasks, out_dir)  

if __name__ == "__main__":
    check_coco(sys.argv[1])