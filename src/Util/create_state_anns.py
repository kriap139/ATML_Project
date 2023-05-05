from cio import clean_dir, dirUp, get_dataset_name, dir_name
import os
from pycocotools.coco import COCO
import cv2
import json
import numpy as np
import sys

def draw_bbox(img: np.ndarray, bbox: tuple, label_text: str, color: tuple):
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(img, (x, y), (x, y + 30), color, -1)
    cv2.putText(img, label_text, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

def create_state_anns(dataset_dir: str, out_dir: str):
    out_img_dir = os.path.join(out_dir, dir_name(dirUp(dataset_dir)), "state_anns", get_dataset_name(dataset_dir))
    clean_dir(out_img_dir)

    ann_file = os.path.join(dataset_dir, "ann.json")
    states_file = os.path.join(dataset_dir, "states.json")
    
    coco = COCO(ann_file)
    states_data = {}

    for img_info in coco.loadImgs(coco.getImgIds()):
        data = {}
        ann_ids = coco.getAnnIds(img_info["id"], coco.getCatIds(supNms=["Valve"]))
        anns = coco.loadAnns(ann_ids)
        
        fn = img_info["file_name"]
        fp = os.path.join(dataset_dir, fn)
        img = cv2.imread(fp)

        for ann in anns:
            draw_bbox(img, ann["bbox"], label_text=f"ann_id={ann['id']}", color=[128, 128, 128, 1])
            data[ann["id"]] = dict(state="OPEN", angle=0.0)
        
        cv2.imwrite(os.path.join(out_img_dir, fn), img)
        states_data[img_info["id"]] = dict(file_name=img_info["file_name"], ann_states=data)
    
    with open(states_file, mode='w') as f:
        json.dump(states_data, f, indent=3)

if __name__ == "__main__":
    create_state_anns(sys.argv[1], sys.argv[2])