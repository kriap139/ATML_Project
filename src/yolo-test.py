import cv2 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import mmcv
import argparse
from config import Test, YOLO_WEIGHTS, YOLO_CFG, OUT_DIR, IMAGE_DIR, BBOX_COLORS, MASK_COLORS, DRAW_ORDER, YOLO_CLASS_MAP
from Util import clean_dir, get_dataset_name
import json
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) yolo model')
    parser.add_argument('dataset_dir', help='directory of coco dataset')
    parser.add_argument('--work-dir', type=str, default=OUT_DIR, help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--test-dataset', help="Dataset to test", type=Test.from_string, choices=list(Test), default=Test.test)
    args = parser.parse_args()
    return args

# Based on previous project
def test_yolo_model(dataset_dir: str, test: Test, out_dir: str, alpha: float = 0.9):
    model = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    layerNames = model.getLayerNames()
    layers = [layerNames[i - 1] for i in model.getUnconnectedOutLayers()]
    
    res_dir = os.path.join(out_dir, f"yolo-{get_dataset_name(dataset_dir)}", str(test))
    clean_dir(res_dir)
    
    coco = COCO(os.path.join(dataset_dir, str(test), 'ann.json'))
    categories = coco.loadCats(coco.getCatIds())
    images = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds()))

    results = []
    for img_info in images:
        fn = img_info["file_name"]
        fp = os.path.join(IMAGE_DIR, fn)

        img = cv2.imread(fp)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        outputs = model.forward(layers)

        classes = []
        confidences = []
        boxes = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append((x, y, w, h))
                    confidences.append(float(confidence))
                    classes.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        joined = list(zip(classes, boxes, confidences))
        check = lambda i, tup: (i in indexes) and (tup[0] in YOLO_CLASS_MAP.keys())
        joined = [tup for i, tup in enumerate(joined) if check(i, tup)]

        #print(f"detections={len(joined)} classes={classes}, confidences={confidences}, indexes={indexes}")

        for cls, bbox, confidence in joined:
             results.append(
                 {
                     "image_id": img_info["id"], 
                     "category_id": YOLO_CLASS_MAP[cls], 
                     "bbox": bbox, 
                     "score": confidence,
                 }
             )
        
        if len(tuple(joined)):
            classes, bboxes, confideces = zip(*joined) 
            classes = [YOLO_CLASS_MAP[cls] for cls in classes]
            classes, bboxes, confideces = np.array(classes), np.array(bboxes), np.array(confideces)
            names = {cat['id']: cat['name'] for cat in coco.loadCats(classes)}

            mmcv.imshow_det_bboxes(fp, bboxes, classes, names, show=False, out_file=os.path.join(res_dir, fn))
        else:
            cv2.imwrite(os.path.join(res_dir, fn), img)

    tmp_fp = os.path.join(tempfile.gettempdir(), 'coco_test.json') 
    with open(tmp_fp, mode='w') as f:
        json.dump(results, f, indent=3)

    coco_res = coco.loadRes(tmp_fp)

    coco_eval = COCOeval(coco, coco_res, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
if __name__ == "__main__":
    args = parse_args()
    test_yolo_model(args.dataset_dir, args.test_dataset, args.work_dir)

    
