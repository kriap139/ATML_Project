from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.evaluation import DumpDetResults
import mmcv
import os
from config import Test, OUT_DIR, IMAGE_DIR, THRESH_ANGLE_CLOSED_DEG, THRESH_ANGLE_OPEN_DEG, FONT
from Util import contour_to_ploygon, make_contours_mask, get_dataset_name
from state_detection import StateDetector, Valve, ValveState, ValveMark, Detection
from Util import clean_dir
import shutil
import json
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import pickle
import numpy as np
import cv2
from typing import List, Tuple
import logging

def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--model', help='model file', type=str)
    parser.add_argument('--work-dir', type=str, default=OUT_DIR, help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--test-dataset', help="Dataset to test", type=Test.from_string, choices=list(Test), default=Test.test)
    parser.add_argument('--skip-state-detect', help="skip state detection test", action='store_true')
    parser.add_argument('--state-detect-only', help="Run only state detect tests on dataset", action='store_true')
    args = parser.parse_args()
    return args


def test_model(config_fp: str, model_fp: str, test: Test, work_dir: str) -> dict:
    cfg = Config.fromfile(config_fp)
    
    out_dir = os.path.join(work_dir, get_dataset_name(cfg.data_root) , str(test))
    res_file = os.path.join(out_dir, "results.pkl")
    clean_dir(out_dir)
    
    cfg.load_from = model_fp
    cfg.work_dir = out_dir
    cfg.default_hooks['visualization'] = dict(type='DetVisualizationHook', draw=True)

    # Fixing paths
    if test == Test.train:
        ann_file = cfg.train_dataloader.get("dataset").get("ann_file")
        data_prefix = cfg.train_dataloader.get("dataset").get("data_prefix")
    elif test == Test.val:
        ann_file = cfg.val_dataloader.get("dataset").get("ann_file")
        data_prefix = cfg.val_dataloader.get("dataset").get("data_prefix")
    elif (cfg.test_dataloader == cfg.val_dataloader):
        ann_file = cfg.test_dataloader.get("dataset").get("ann_file")
        ann_file = os.path.join('test', os.path.basename(ann_file))
        data_prefix = dict(img='test/')
        
    if (test in (Test.train, Test.val)) or (cfg.test_dataloader == cfg.val_dataloader):
        cfg.test_dataloader.get("dataset")["ann_file"] = ann_file
        cfg.test_dataloader.get("dataset")["data_prefix"].update(data_prefix)
        cfg.test_evaluator["ann_file"] = os.path.join(cfg.data_root, ann_file)
    
    runner = Runner.from_cfg(cfg)
    file_dumper = DumpDetResults(out_file_path=res_file)
    runner.test_evaluator.metrics.append(file_dumper)
    return runner.test()

def test_state_detect(config_fp: str, test: Test, work_dir: str):
    cfg = Config.fromfile(config_fp)

    out_dir = os.path.join(work_dir, get_dataset_name(cfg.data_root) , str(test), "state_detect")
    out_img_dir = os.path.join(out_dir, "images")
    out_img_mask_dir = os.path.join(out_dir, "mask_images")
    out_file = os.path.join(out_dir, "state_metrics.json")
    clean_dir(out_dir)
    clean_dir(out_img_dir)
    clean_dir(out_img_mask_dir)
    
    res_file = os.path.join(work_dir, get_dataset_name(cfg.data_root) , str(test), "results.pkl")
    ann_file = os.path.join(cfg.data_root, str(test), "ann.json")
    
    coco = COCO(ann_file)
    with open(res_file, mode='rb') as f:
        results = pickle.load(f, encoding='utf-8')
    
    detector = StateDetector(THRESH_ANGLE_CLOSED_DEG, THRESH_ANGLE_OPEN_DEG, FONT)

    valves_by_img = {} 
    gts_by_img = detector.load_state_ground_truths(coco, os.path.join(cfg.data_root, str(test), "states.json"))

    for count, result in enumerate(results):
        fn = coco.loadImgs(result["img_id"])[0]["file_name"]
        n_iter = count + 1

        # 'masks', 'labels', 'bboxes', 'scores'
        preds = result["pred_instances"]
        valves = []
        marks = []

        for i in range(len(preds['masks'])):
            mask = maskUtils.decode(preds["masks"][i])
            rle = preds["masks"][i]
            bbox = [int(val) for val in preds['bboxes'][i]]  
            label = int(preds["labels"][i]) + 1 # fixme: label starts from 0!
            score = float(preds['scores'][i])
            cat = coco.loadCats(label)[0] 

            if cat['supercategory'] == 'Mark':
                marks.append(ValveMark(label, cat["name"], bbox, mask, rle, score))
            elif cat['supercategory'] == 'Valve':
                valves.append(Valve(label, cat['name'], bbox, mask, rle, score))
            else:
                raise ValueError(f"Unsupported category {cat['name']}")

        ious_per_valve = []
        for valve in valves:
            ious = [maskUtils.iou([mark.rle], [valve.rle], [0])[0][0] for mark in marks]
            ious_per_valve.append(ious)
        
        ious_per_valve = np.array(ious_per_valve)
        ious_per_mark = np.transpose(ious_per_valve)
        sorted_best_idx_valve = ious_per_valve.argsort(axis=1)[::-1]
        sorted_best_idx_mark = ious_per_mark.argsort(axis=0)[::-1]

        assert len(ious_per_valve) == len(valves)
        assert len(ious_per_mark) == len(marks)
        
        for valve_idx, valve in enumerate(valves):
            for mark_idx in sorted_best_idx_valve[valve_idx]:
                iou = ious_per_valve[valve_idx][mark_idx]
                mark: ValveMark = marks[mark_idx]

                if valve_idx == sorted_best_idx_mark[mark_idx][0]:
                    if (iou > 0) and (f"{valve.cls_name}_Mark" == mark.cls_name):
                        valve.mark = mark
                        marks[mark_idx] = None
                        break
        
        orphans = [valve for valve in valves if valve.mark is None]
        orphans.extend(mark for mark in marks if mark is not None)
        valves = list(filter(lambda v: v.mark is not None, valves))

        valve_ious = [f"v{i}={list(np.round(ious, 4))}" for i, ious in enumerate(ious_per_valve)]
        print(f"{n_iter}: n_valves={len(marks)}, n_marks={len(marks)}, valves={len(valves)}, orphans={len(orphans)}\tious: {', '.join(valve_ious)}")

        valves_by_img[result["img_id"]] = valves

        fp = os.path.join(cfg.data_root, str(test), fn)
        img = cv2.imread(fp)

        detector.stateDetect_bbox(valves)
        detector.draw(img, valves)    
        for orphan in orphans:
            detector.draw_bbox(img, orphan, is_orphan=True)
    
        for jj, valve in enumerate(valves):
            make_contours_mask([valve.mask_contour], img.shape, 
                               save_fp=os.path.join(out_img_mask_dir, f"{n_iter}_v{jj + 1}_{fn}"))
            make_contours_mask([valve.mark.mask_contour], img.shape, (145, 145, 145), 
                               save_fp=os.path.join(out_img_mask_dir, f"{n_iter}_v{jj + 1}_mark_{fn}"))
            
        cv2.imwrite(os.path.join(out_img_dir, f"{n_iter}_{fn}"), img)
    
    metrics = detector.calc_metrics(valves_by_img, gts_by_img)
    print(json.dumps(metrics, indent=3))
    with open(out_file, mode='w') as f:
        json.dumps(metrics, indent=3) 



if __name__ == "__main__":
    logging.getLogger().setLevel(0)
    args = parse_args()

    if args.state_detect_only:
        test_state_detect(args.config, args.test_dataset, args.work_dir)
        exit(0)

    if args.model is None:
        raise RuntimeError("--model param required for testing model")
    
    metrics = test_model(args.config, args.model, args.test_dataset, args.work_dir)
    print(json.dumps(metrics, indent=3))

    if not args.skip_state_detect:
        test_state_detect(args.config, args.test_dataset, args.work_dir)
