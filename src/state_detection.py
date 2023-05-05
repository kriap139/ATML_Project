import numpy as np
import cv2
from typing import Tuple, Union, Dict, List
from config import ValveState, Number, MASK_COLORS, STATE_COLORS_BGRA, BBOX_COLORS, STATE_VEC_COLORS_BGR, CATEGORY_IDS
from Util import merge_contours
import mmcv
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class Detection:
    def __init__(self,
                    class_id: int, 
                    class_name: str, 
                    bbox: np.ndarray,
                    mask: np.ndarray,
                    rle: dict,
                    confidence: float,
                    line: tuple = None):
        
        self.cls_id = class_id
        self.cls_name = class_name
        self.bbox = bbox
        self.mask = mask
        self.rle = rle
        self.confidence = confidence
        self.line = line
        self.vec = None
        self.mask_contour = None

class ValveMark(Detection):
    def __init__(self,
                    class_id: int, 
                    class_name: str, 
                    bbox: np.ndarray,
                    mask: np.ndarray,
                    rle: dict,
                    confidence: float,
                    line: tuple = None):
        super(ValveMark, self).__init__(class_id, class_name, bbox, mask, rle, confidence, line)


class Valve(Detection):
    def __init__(self, 
                 class_id: int, 
                 class_name: str, 
                 bbox: np.ndarray,
                 mask: np.ndarray,
                 rle: dict,
                 confidence: float,
                 mark: ValveMark = None,
                 line: tuple = None,
                 state: ValveState = ValveState.UNKNOWN, 
                 angle: float = None):
        
        super(Valve, self).__init__(class_id, class_name, bbox, mask, rle, confidence, line)
        self.mark = mark
        self.state = state
        self.angle = angle

class StateDetector:
    def __init__(self, angleClosedThreshDeg: float = 70, angleOpenThreshDeg: float = 19, font = cv2.FONT_HERSHEY_PLAIN):
        self.angleClosedThreshDeg = angleClosedThreshDeg
        self.angleOpenThreshDeg = angleOpenThreshDeg
        self.font = font
    
    def load_state_ground_truths(self, coco: COCO, states_fp: str) -> List[Valve]:
        gts = {}

        with open(states_fp, mode='r') as f:
            states_data = json.load(f)

        for img_info in coco.loadImgs(coco.getImgIds()):
            ann_ids = coco.getAnnIds(img_info["id"], coco.getCatIds(supNms=["Valve"]))
            ann_states = states_data[str(img_info["id"])]["ann_states"]
            valves = []

            for ann in coco.loadAnns(ann_ids):
                ann_id = str(ann["id"])
                state, angle = ann_states[ann_id]["state"], ann_states[ann_id]["angle"]
                cat  = coco.loadCats(ann["category_id"])[0]
                x, y, w, h = ann["bbox"]

                cls_id = ann["category_id"]
                cls_name = cat["name"]
                bbox = [x, y, x + w, y + h]
                mask = coco.annToMask(ann)
                rle = coco.annToRLE(ann)

                valves.append(Valve(cls_id, cls_name, bbox, mask, rle, 0, state=ValveState[state], angle=angle))

            gts[img_info["id"]] = valves
        return gts

    def calc_metrics(self, valves_by_img: Dict[int, List[Valve]], gts_by_img: Dict[int, List[Valve]]): 
        y_true = []
        y_pred = []
        orphan_gts = []

        
        for img_id, valves in valves_by_img.items():
            gts = gts_by_img[img_id]
            ious_per_valve = []

            for valve in valves:
                ious = [maskUtils.iou([valve.rle], [gt.rle], [0])[0][0] for gt in gts]
                ious_per_valve.append(ious)
            
            ious_per_valve = np.array(ious_per_valve)
            ious_per_gt = np.transpose(ious_per_valve)
            sorted_best_idx_valve = ious_per_valve.argsort(axis=1)[::-1]
            sorted_best_idx_gt = ious_per_gt.argsort(axis=1)[::-1]

            assert len(ious_per_valve) == len(valves)
            assert len(ious_per_gt) == len(gts)
            
            for valve_idx, valve in enumerate(valves):
                for gt_idx in sorted_best_idx_valve[valve_idx]:
                    iou = ious_per_valve[valve_idx][gt_idx]
                    gt: Valve = gts[gt_idx]

                    if valve_idx == sorted_best_idx_gt[gt_idx][0]:
                        if (iou > 0) and (valve.cls_id == gt.cls_id):
                            y_true.append(gt)
                            y_pred.append(valve)
                            gts[gt_idx] = None
                            break
            
            orphan_gts.extend(filter(lambda gt: gt is not None, gts))
        
        metrics_by_states = {
            "OPEN": dict(missed=0, correct=0, wrong=0, angle_errors=[]),
            "CLOSED": dict(missed=0, correct=0, wrong=0, angle_errors=[]),
            "PARTIAL": dict(missed=0, correct=0, wrong=0, angle_errors=[])
        }

        for orphan in orphan_gts:
            metrics_by_states[orphan.state.name]["missed"] += 1
        
        for i, p in enumerate(y_pred):
            pt = y_true[i]
            key = 'correct' if p.state == pt.state else 'wrong'
            metrics_by_states[pt.state.name][key] += 1
            metrics_by_states[pt.state.name]["angle_errors"].append(abs(pt.angle - p.angle))
        
        metrics = {
            "OPEN": dict(n=0, missed=0, correct=0, wrong=0, acc_found=0, acc=0, mean_err=0.0, std_err=0),
            "CLOSED": dict(n=0, missed=0, correct=0, wrong=0, acc_found=0, acc=0, mean_err=0.0, std_err=0),
            "PARTIAL": dict(n=0, missed=0, correct=0, wrong=0, acc_found=0, acc=0, mean_err=0.0, std_err=0),
            "TOTAL": dict(n=0, missed=0, correct=0, wrong=0, acc_found=0, acc=0, mean_err=0.0, std_err=0)
        }

        #print(json.dumps(metrics_by_states, indent=3))

        for state, data in metrics_by_states.items():
            missed, correct, wrong, errs = data["missed"], data["correct"], data["wrong"], data["angle_errors"]

            metrics[state]["n"] = missed + correct + wrong
            metrics[state]["missed"] = missed
            metrics[state]["correct"] = correct
            metrics[state]["wrong"] = wrong
            metrics[state]["acc_found"] = float(correct) / max(1, (correct + wrong))
            metrics[state]["acc"] = float(correct) / max(1, (correct + wrong + missed))
            metrics[state]["mean_err"] = float(np.mean(errs) if len(errs) else 0)
            metrics[state]["std_err"] = float(np.std(errs) if len(errs) else 0)
        
        missed = sum(d["missed"] for d in metrics_by_states.values())
        correct = sum(d["correct"] for d in metrics_by_states.values())
        wrong = sum(d["wrong"] for d in metrics_by_states.values())
        
        errors = [] 
        for d in metrics_by_states.values():
            errors.extend(d["angle_errors"])

        total = metrics["TOTAL"]
        total["n"] = missed + correct + wrong
        total["missed"] = missed
        total["correct"] = correct
        total["wrong"] = wrong
        total["acc_found"] = float(correct) / (correct + wrong)
        total["acc"] = float(correct) / (correct + wrong + missed)
        total["mean_err"] = float(np.mean(errors))
        total["std_err"] = float(np.std(errors))

        return metrics
        
    @staticmethod
    def map(val: Number, inMin: Number, inMax: Number, outMin: Number, outMax: Number) -> float:
        return ((val - inMin) * (outMax - outMin) / float(inMax - inMin)) + outMin

    def calcState(self, vec_pipe: np.ndarray, vec_mark: np.ndarray) -> Tuple[ValveState, float]:
        dot = np.dot(np.transpose(vec_mark), vec_pipe)
        angle, = np.arccos(dot)
        deg,  = np.abs(np.degrees(angle))

        if deg > 90:
            newDeg = self.map(val=deg, inMin=90, inMax=180, outMin=90, outMax=0)
        else:
            newDeg = deg

        if newDeg > self.angleClosedThreshDeg:
            return ValveState.CLOSED, deg
        elif newDeg < self.angleOpenThreshDeg:
            return ValveState.OPEN, deg
        else:
            return ValveState.PARTIAL, deg
    
    def draw_vec(self, img: np.ndarray, bbox: tuple, line: tuple, color: tuple):
        rows, cols = img.shape[:2]

        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        bh, bw = crop.shape[:2]

        if (bh == 0) or (bw == 0):
            return
        
        (vx, vy, x, y) = line
        lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

        if (np.abs(lefty) < rows) and (np.abs(righty) < rows):
            cv2.arrowedLine(img, (0, lefty), (rows - 1, righty), color, thickness=3, tipLength=0.03)
    
    def draw_vecs(self, img: np.ndarray, valve: Valve):
       self.draw_vec(img, valve.bbox, valve.line, STATE_VEC_COLORS_BGR[valve.cls_id])
       self.draw_vec(img, valve.mark.bbox, valve.mark.line, STATE_VEC_COLORS_BGR[valve.mark.cls_id])

    def draw_bbox(self, img: np.ndarray, obj: Detection, is_orphan=False, label_mark=False, label_text: str = None):
        x1, y1, x2, y2 = obj.bbox
            
        if is_orphan:
            label = f"Orphan"
            color = STATE_COLORS_BGRA.get(ValveState.UNKNOWN)
        elif type(obj) == Valve:
            color = STATE_COLORS_BGRA.get(obj.state)
            label = f"{obj.cls_name}, angle={np.round(obj.angle, 3)}" 
        elif type(obj) == ValveMark:
            color = BBOX_COLORS.get(obj.cls_id)
            label = f"Mark"
        else:
            color = STATE_COLORS_BGRA.get(ValveState.UNKNOWN)
        
        if label_text is not None:
            label = label_text
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if (type(obj) == ValveMark and label_mark) or isinstance(obj, Detection):
            cv2.rectangle(img, (x1, y1), (x2, y1 + 30), color, -1)
            cv2.putText(img, label, (x1, y1 + 20), self.font, 1.5, (255, 255, 255), 2)
    
    def draw_mask(self, img: np.ndarray, valve: Valve, draw_mark=True, alpha: float = 0.9):
        overlay = img.copy()
        
        contours = cv2.findContours(valve.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        color = MASK_COLORS[valve.cls_id]
        cv2.drawContours(overlay, contours, -1, color, -1)

        if draw_mark:
            contours = cv2.findContours(valve.mark.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            color = MASK_COLORS[valve.mark.cls_id]
            cv2.drawContours(overlay, contours, -1, color, -1)
            
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        

    def create_line(self, points: list):
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        (vx, vy, x, y) = line
        vec = np.array((vx, vy))
        return line, vec
    
    def draw(self, img: np.ndarray, valves: List[Valve], draw_masks = True, draw_bboxes = True, draw_mark_bboxes = True):
        for valve in valves:
            if draw_masks:
                self.draw_mask(img, valve)
            if draw_bboxes:
                self.draw_bbox(img, valve)
            if draw_mark_bboxes:
                self.draw_bbox(img, valve.mark)
            self.draw_vecs(img, valve)
    
    def stateDetect_bbox(self, valves: List[Valve]):
        for valve in valves:
            mark = valve.mark

            contour = self.process_mask(mark.mask)
            mar = cv2.minAreaRect(contour)
            points = cv2.boxPoints(mar)
            mark.line, mark.vec = self.create_line(points)
            mark.mask_contour = contour

            contour = self.process_mask(valve.mask)
            mar = cv2.minAreaRect(contour)
            points = cv2.boxPoints(mar)
            valve.line, valve.vec = self.create_line(contour)
            valve.mask_contour = contour

            valve.state, valve.angle = self.calcState(valve.vec, mark.vec)

    def process_mask(self, mask: np.ndarray):
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        merged = merge_contours(contours)
        return merged

    def stateDetect_mask(self, valves: List[Valve]):
        for valve in valves:
            mark = valve.mark

            contour = self.process_mask(mark.mask)
            mark.line, mark.vec = self.create_line(contour)
            mark.mask_contour = contour

            contour = self.process_mask(valve.mask)
            valve.line, valve.vec = self.create_line(contour)

            valve.state, valve.angle = self.calcState(valve.vec, mark.vec)
            valve.mask_contour = contour
            