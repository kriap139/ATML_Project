import glob
import json
import os
import cv2
import sys
from typing import Dict
import numpy as np
from Util.cio import dirUp
from Util.drawing import save_draw_contours, show_draw_contours
import logging

#from PIL import Image                                      # (pip install Pillow)
#import numpy as np                                         # (pip install numpy)
#from skimage import measure                                # (pip install scikit-image)
#from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)


# Taken form: https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data/58874392#58874392
def sort_xy(x, y) -> tuple:

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    return x_sorted, y_sorted

# modified code from: https://github.com/brunobelloni/binary-to-coco-json-converter

def get_coco_json_format():
    return {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }


def find_contours(sub_mask):
    gray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


def create_category_annotation(category_ids: dict, super_categories: dict):
    category_list = []
    for key, value in category_ids.items():
        category = {"id": value, "name": key, "supercategory": super_categories.get(key)}
        category_list.append(category)
    return category_list


image_id = 0
def create_image_annotation(file_name, width, height):
    global image_id
    image_id += 1
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }

def contour_to_ploygon(contour: np.ndarray) -> np.ndarray:
    reshaped = contour.reshape(-1, 2)
    # print(f"contour: {contour.shape}, reshaped: {reshaped.shape}")
    return reshaped.ravel(order='C')

def create_annotation_format(mask_fn: str, contours, image_id_, category_id, annotation_id) -> dict:
    ann = {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "area": 0,
        "segmentation": []
    }

    if len(contours) == 1:
        ann["bbox"] = cv2.boundingRect(contours[0])
        ann["area"] = cv2.contourArea(contours[0])

        ann["segmentation"].append(contour_to_ploygon(contours[0]).tolist())
    elif len(contours) <= 0:
        print(f"No contours found in mask: {mask_fn}")
    else:
        points = [] 
        for contour in contours:
            points += [pt[0] for pt in contour]

        points = np.array(points)
        assert points.shape[1] == 2
        x, y = points.T
        x, y = sort_xy(x, y)

        points = np.stack((x,y), axis = 1)
        contour = np.array(points).reshape((-1,1,2)).astype(np.int32)
        ann["bbox"] = cv2.boundingRect(contour)

        for cnt in contours:
            ann["segmentation"].append(contour_to_ploygon(cnt).tolist())
            ann["area"] += cv2.contourArea(cnt)
        
    return ann

def images_annotations_info(masks_dir: str, info_file: str, category_ids: dict, image_dir: str = None, raw_test_dir: str = None):
    annotation_id = 0
    annotations = []
    images: Dict[str, dict] = {}
    images_fns: Dict[str, bool] = {}
    count_no_img = 0

    for img_fp in glob.glob(os.path.join(image_dir, '*')):
        fn = os.path.basename(img_fp)
        images_fns[fn] = True

    with open(info_file, mode='r') as f:
        data = json.load(f)
        info = {d["id"]: os.path.basename(d["image"]) for d in data}

    for mask_fp in glob.glob(os.path.join(masks_dir, '*')):
        mask_fn = os.path.basename(mask_fp)
        
        # task-1904-annotation-2-by-1-tag-Valve_Lever_Brown-0.png
        parts = mask_fn.split("-")
        assert parts[0] == "task"
        task_id = int(parts[1])
        category = parts[parts.index("tag") + 1]
        image_fn = info[task_id]

        if not images_fns.get(image_fn, False):
            count_no_img += 1
            continue

        mask_img = cv2.imread(mask_fp)
        height, width, c = mask_img.shape

        image = images.get(image_fn, None)

        if image is None:
            image = create_image_annotation(file_name=image_fn, width=width, height=height)
            images[image_fn] = image

        contours = find_contours(mask_img)
     
        if raw_test_dir is not None:
            save_draw_contours(image_dir, image_fn, raw_test_dir, contours, (0, 255, 0))

        print(f"task_id: {task_id}, category: {category}, image_fn: {image_fn}, n_contours: {len(contours)}")
        annotation = create_annotation_format(mask_fn, contours, image['id'], category_ids[category], annotation_id)

        if annotation['area'] > 0:
            annotations.append(annotation)
            annotation_id += 1
        else:
            print(f"Annotation area less than 0 for mask: {mask_fn}")
    
    if count_no_img > 0:
        print(f"{count_no_img} annotations skipped: missing images in image_dir ({image_dir})")

    return list(images.values()), annotations, annotation_id, info

def mask_draw_test(coco: dict, image_dir: str, output_dir: str, colors: dict, bbox_colors: dict, draw_orders: dict = None, alpha: float = 0.9):
    os.makedirs(output_dir, exist_ok=True)
    images = {d["id"]: d for d in coco["images"]}
    contours_by_img = {d["id"]: [] for d in coco["images"]}
    bboxes_by_img = {d["id"]: [] for d in coco["images"]}
    contour_category_by_img = {d["id"]: [] for d in coco["images"]}
    bbox_category_by_img = {d["id"]: [] for d in coco["images"]}

    for ann in coco["annotations"]:
        contours = contours_by_img[ann["image_id"]]
        bboxes = bboxes_by_img[ann["image_id"]]
        cnts_category_ids = contour_category_by_img[ann["image_id"]]
        bbox_category_ids = bbox_category_by_img[ann["image_id"]]

        bboxes.append(ann["bbox"])
        bbox_category_ids.append(ann["category_id"])
        
        for seg in ann["segmentation"]:
            contour = np.array(seg, dtype=np.int32).reshape((-1, 2), order='C')
            contours.append(contour)
            cnts_category_ids.append(ann["category_id"])
        
    
    for i, (img_id, contours) in enumerate(contours_by_img.items()):
        fn = images[img_id]["file_name"]
        
        img = cv2.imread(os.path.join(image_dir, fn))
        overlay = img.copy()

        # draw contours
        if draw_orders is not None:
            joined = list(zip(contours, contour_category_by_img[img_id]))
            joined.sort(key=lambda tup: draw_orders[tup[1]])
            contours, contour_category_by_img[img_id] = zip(*joined)


        for i, contour in enumerate(contours):
            category_id = contour_category_by_img[img_id][i]
            color = colors[category_id]
            cv2.drawContours(overlay, [contour], -1, color, -1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        for i, bbox in enumerate(bboxes_by_img[img_id]):
            x, y, w, h = bbox
            category_id = bbox_category_by_img[img_id][i]
            bbox_color = bbox_colors[category_id]
            cv2.rectangle(img, (x, y), (x + w, y + h), bbox_color, 3)
        
        
        cv2.imwrite(os.path.join(output_dir, fn), img)

def coco_annotations_from_masks(root_dir: str, category_ids: dict, super_categories: dict, image_dir: str, mask_colors: dict = None, bbox_colors: dict = None, draw_orders: dict = None, create_tests=True):
    """Root dir needs to have a sub dir called masks where the image masks are stored, and the json-mini file for identifying image names. 
        This data is exported from label studio."""
    
    
    masks_dir = os.path.join(root_dir, "masks")
    info_file = os.path.join(root_dir, "json-mini.json")
    ann_file = os.path.join(root_dir, "ann.json")
    img_mapping_file = os.path.join(root_dir, "image_task_mappings.json")
    test_dir = os.path.join(root_dir, "tests")
    raw_test_dir = os.path.join(root_dir, "raw_tests")

    coco = get_coco_json_format()
    
    # Create category section
    coco["categories"] = create_category_annotation(category_ids, super_categories)

    # Create images and annotations sections
    if create_tests:
        coco["images"], coco["annotations"], annotation_cnt, img_info = images_annotations_info(masks_dir, info_file, category_ids, image_dir, raw_test_dir)
    else:
        coco["images"], coco["annotations"], annotation_cnt, img_info = images_annotations_info(masks_dir, info_file, category_ids)

    with open(ann_file, "w") as f:
        json.dump(coco, f, sort_keys=True, indent=4)
    
    with open(img_mapping_file, mode='w') as f:
        json.dump(img_info, f, indent=3)
    
    print(f"Created {annotation_cnt} annotations for image_masks in folder: {masks_dir}")

    if create_tests:
        if (bbox_colors is None) or (mask_colors is None):
            logging.warn("bbox and/or mask colors not specified, skipping mask tests")
        else:
            print(f"Creating test images in: {test_dir}")
            mask_draw_test(coco, image_dir, test_dir, mask_colors, bbox_colors, draw_orders)
    
    return coco