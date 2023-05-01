
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Union, Dict, List
from dataclasses import dataclass
from enum import Enum, unique

Number = Union[float, int]

@unique
class ValveState(Enum):
    UNKNOWN = 1
    OPEN = 2
    CLOSED = 3
    PARTIAL = 4

class Valve:
    STATE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self, 
                 class_id: int, 
                 class_name: str, 
                 bbox: np.ndarray,
                 mask: np.ndarray,
                 bbox_mark: np.ndarray,
                 mask_mark: np.ndarray, 
                 confidence: float,
                 state: ValveState = ValveState.UNKNOWN, 
                 angle: float = None):
        
        self.cls_id = class_id
        self.cls_name = class_name
        self.state = state
        self.angle = angle
        self.bbox = bbox
        self.mask = mask
        self.bbox_mark = bbox_mark
        self.mask_mark = mask_mark
        self.confidence = confidence
        self.line_mark = None
        self.line_pipe = None

    def get_state_color(self) -> tuple:
        return self.STATE_COLORS_BGRA.get(self.state)


class StateDetector:
    def __init__(self, dataset_dir, results: dict, angleClosedThreshDeg: float = 70, angleOpenThreshDeg: float = 19):
        self.angleClosedThreshDeg = angleClosedThreshDeg
        self.angleOpenThreshDeg = angleOpenThreshDeg
        self.font = cv2.FONT_HERSHEY_PLAIN
    
    @staticmethod
    def map(val: Number, inMin: Number, inMax: Number, outMin: Number, outMax: Number) -> float:
        return ((val - inMin) * (outMax - outMin) / float(inMax - inMin)) + outMin

    def calcState(self, angle: float) -> ValveState:
        deg = np.abs(np.degrees(angle))

        if deg > 90:
            newDeg = self.map(val=deg, inMin=90, inMax=180, outMin=90, outMax=0)
        else:
            newDeg = deg

        if newDeg > self.angleClosedThreshDeg:
            return ValveState.CLOSED
        elif newDeg < self.angleOpenThreshDeg:
            return ValveState.OPEN
        else:
            return ValveState.PARTIAL
    
    def draw_vecs(self, img: np.ndarray, valve: Valve):
        h, w = img.shape[:2]
        x, y, w, h = valve.bbox
        
        crop = img[y:y + h, x:x + w]
        bh, bw = crop.shape[:2]

        if (crop.shape[0] == 0) or (crop.shape[1] == 0):
            return
        
        (vx, vy, x, y) = valve.line_mark
        lefty, righty = int((-x * vy / vx) + y), int(((bw - x) * vy / vx) + y)

        if (np.abs(lefty) < h) and (np.abs(righty) < h):
            cv2.arrowedLine(img, (0, lefty), (bw - 1, righty), (0, 223, 255), thickness=6, tipLength=0.03)
        
        (vx, vy, x, y) = valve.line_pipe
        lefty, righty = int((-x * vy / vx) + y), int(((bw - x) * vy / vx) + y)

        if (np.abs(lefty) < h) and (np.abs(righty) < h):
            cv2.arrowedLine(img, (0, lefty), (bw - 1, righty), (0, 223, 255), thickness=6, tipLength=0.03)

    def draw(self, img: np.ndarray, valve: Valve):
        x, y, w, h = valve.bbox

        color = valve.get_state_color()
        label = f"cls={valve.cls_id}, angle={valve.angle}"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + 30), color, -1)
        cv2.putText(img, label, (x, y + 30), self.font, 2, (255, 255, 255), 3)

    def stateDetect(self, img: np.ndarray, valves: List[Valve], draw: bool = True):
        # Deleting noises which are in area of mask
        #mask = cv2.erode(mask, None, iterations=2)
        #mask = cv2.dilate(mask, None, iterations=2)
        # define kernel size
        #kernel = np.ones((5, 5), np.uint8)
        # Remove unnecessary noise from mask
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        for valve in valves:
            valve.line_mark =cv2.fitLine(valve.mask_mark, cv2.DIST_L12, 0, 0.01, 0.01)
            (vx, vy, x, y) = valve.line_mark
            vec_mark = np.array((vx, vy))
            
            valve.line_pipe = cv2.fitLine(valve.mask, cv2.DIST_L12, 0, 0.01, 0.01)
            (vx, vy, x, y) = valve.line_mark
            vec_pipe = np.array((vx, vy))

            dot = np.dot(np.transpose(vec_mark), vec_pipe)
            angle, = np.arccos(dot)

            state = self.calcState(angle)
            valve.angle = np.degrees(angle)
            valve.state = state

            if draw:
                self.draw(img, valve)