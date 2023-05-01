import cv2 
from pycocotools.coco import COCO
import os
import numpy as np
import mmcv

# Based on previous project
def test_yolo_model(weights_fp, cfg_fp: str, dataset_dir: str, image_dir: str, 
                    cat_draw_order: dict, colors_by_cat: dict, out_dir: str, alpha: float = 0.9):
    
    model = cv2.dnn.readNet(weights_fp, cfg_fp)
    layerNames = model.getLayerNames()
    layers = [layerNames[i - 1] for i in model.getUnconnectedOutLayers()]
    tag_cls_id = 8

    coco = COCO(os.path.join(dataset_dir, 'ann.json'))
    categories = coco.loadCats(coco.getCatIds())
    labels = [category['name'] for category in categories]
    images = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds()))

    results = {im["id"] for im in images}
    
    for img_info in images:
        fn = img_info["file_name"]
        fp = os.path.join(image_dir, fn)

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
        indexes = [i for i in indexes if i != tag_cls_id]

        joined = zip(classes, boxes, confidence)
        joined = filter(lambda tup: tup[0] in indexes)
        
        for cls, confidence, bbox in joined:
             results[img_info["id"]].append(
                 {
                     "image_id": img_info["id"], 
                     "category_id": cls, 
                     "bbox": bbox, 
                     "score": confidence,
                 }
             )

        # mmcv.imshow_bboxes()
        


        # ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=coco.getCatIds(), iscrowd=None)
        # anns = coco.loadAnns(ann_ids)
        # anns.sort(key=lambda ann: cat_draw_order[ann["category_id"]])
        
        # img = cv2.imread(fp)
        # overlay = img.copy()

        # for ann in anns:
        #     mask = coco.annToMask(ann)
        #     contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        #     color = colors_by_cat[ann["category_id"]]
        #     cv2.drawContours(overlay, contours, -1, color, -1)
            
        # cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # cv2.imwrite(os.path.join(out_dir, fn), img)
    


    
