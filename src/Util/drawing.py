import cv2
import os
import numpy as np

def show_draw_contours(image_dir: str, image_fn: str, contours: np.ndarray, color: tuple):
    img = cv2.imread(os.path.join(image_dir, image_fn))
    cv2.drawContours(img, contours, -1, color, 2)
    cv2.imshow(f"contours for {image_fn}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_draw_contours(image_dir: str, image_fn: str, save_dir: str, contours: np.ndarray, color: tuple):
    os.makedirs(save_dir, exist_ok=True)
    save_fp = os.path.join(save_dir, image_fn)

    if os.path.exists(save_fp):
        img = cv2.imread(save_fp)
    else:
        img = cv2.imread(os.path.join(image_dir, image_fn))

    cv2.drawContours(img, contours, -1, color, 2)
    cv2.imwrite(save_fp, img)