import sys
import os
import tempfile
import json
from Util import (
    coco_annotations_from_masks, create_train_test_split, random_coco_val_images,
    create_coco_dataset, check_coco, coco_draw_masks_test
)
from config import PROC_DIR, CATEGORY_IDS, SUPER_CATEGORIES, MASK_COLORS, BBOX_COLORS, DRAW_ORDER, IMAGE_DIR

def create_datasets(coco: dict, image_dir: str, out_dir: str, val_samples: float = 0.10, test_samples: float = None):
    val_names = random_coco_val_images(coco, n_samples=val_samples)
    coco_train, coco_val = create_train_test_split(coco, val_names)
    coco_test = None

    if test_samples is not None:
        test_names = random_coco_val_images(coco, n_samples=val_samples)
        coco_train, coco_test = create_train_test_split(coco, test_names)

    create_coco_dataset(coco_train, os.path.join(out_dir, 'train'), image_dir, overwrite=True)
    create_coco_dataset(coco_val, os.path.join(out_dir, 'val'), image_dir, overwrite=True)

    if coco_test is not None:
        create_coco_dataset(coco_test, os.path.join(out_dir, 'test'), image_dir, overwrite=True)

if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]
    coco = coco_annotations_from_masks(ROOT_DIR, CATEGORY_IDS, SUPER_CATEGORIES, IMAGE_DIR, MASK_COLORS, BBOX_COLORS, DRAW_ORDER, create_tests=True)
    check_coco(os.path.join(ROOT_DIR, "ann.json"))
    
    tmp_fp = os.path.join(tempfile.gettempdir(), 'coco_test.json') 
    with open(tmp_fp, mode='w') as f:
        json.dump(coco, f, indent=3)
    coco_draw_masks_test(tmp_fp, IMAGE_DIR, DRAW_ORDER, MASK_COLORS, os.path.join(ROOT_DIR, 'coco_draw_test'))


    if len(sys.argv) == 3:
        OUT_DIR = sys.argv[2]
        create_datasets(coco, IMAGE_DIR, OUT_DIR, val_samples=0.10, test_samples=0.10)



   


