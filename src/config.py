from enum import Enum
from Util.cio import dirUp, clean_dir
import os
from typing import Tuple, Union
from enum import Enum, unique
import cv2

Number = Union[float, int]

class Test(Enum):
    train = 1
    val = 2
    test = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return Test[s]
        except KeyError:
            raise ValueError()


@unique
class ValveState(Enum):
    UNKNOWN = 1
    OPEN = 2
    CLOSED = 3
    PARTIAL = 4


STATE_COLORS_BGRA = {
    ValveState.UNKNOWN: [128, 128, 128, 1],
    ValveState.CLOSED: [10, 34, 224, 1],
    ValveState.OPEN: [0, 230, 17, 1],
    ValveState.PARTIAL: [2, 210, 238, 1]
}


THRESH_ANGLE_CLOSED_DEG = 75
THRESH_ANGLE_OPEN_DEG = 15
FONT = cv2.FONT_HERSHEY_PLAIN


PROC_DIR = dirUp(__file__, 2) 

CATEGORY_IDS = {
    "Valve_Lever_Brown": 1,
    "Valve_Lever_Brown_Mark": 2
}

SUPER_CATEGORIES = {
    "Valve_Lever_Brown": "Valve", 
    "Valve_Lever_Brown_Mark": "Mark"
}

MASK_COLORS = {
    1: (88, 125, 217),
    2: (25,140,168)
}

BBOX_COLORS = {
    1: (0, 0, 255),
    2: (255, 0, 0)
}

DRAW_ORDER = {
    1: 0,
    2: 1
}

YOLO_CLASS_MAP = {
    3: 1
}

STATE_VEC_COLORS_BGR = {
    1: (245, 245, 245), # Valve: Whitesmoke 
    2: (240, 32, 160) # Mark: Purple
}

YOLO_WEIGHTS = os.path.join(PROC_DIR, "data", "models", "yolo", "yolo_tiny.weights")
YOLO_CFG = os.path.join(PROC_DIR, "data", "models", "yolo", "yolo_tiny.cfg")

IMAGE_DIR = os.path.join(PROC_DIR, "data", "dataset", "mask-rcnn")
OUT_DIR = os.path.join(PROC_DIR, "data", "results")


os.makedirs(OUT_DIR, exist_ok=True)







