import os
import sys
import cv2
import numpy as np
import torch

# Add workspace root to sys.path for detection_yolox import
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_ROOT)
# Add detection_yolox to path
DETECTION_YOLOX_PATH = os.path.join(WORKSPACE_ROOT, 'detection_yolox')
sys.path.insert(0, DETECTION_YOLOX_PATH)

from yolo import YOLO

class DeepSORVFYOLOXInfer:
    def __init__(self, config_path=None, weights_path=None, device='cpu', conf_thres=0.3):
        # config_path and weights_path are not used directly, but kept for API compatibility
        self.model = YOLO()
        self.conf_thres = conf_thres

    def infer(self, image):
        # image: np.ndarray (BGR)
        # Returns: list of [x1, y1, x2, y2, score, class_id]
        # The DeepSORVF YOLO returns results as a list of dicts
        results = self.model.detect_image(image, crop=False, count=False, return_raw=True)
        # results: list of dicts with keys: 'box' (x1, y1, x2, y2), 'score', 'class_id', 'class_name'
        boxes = []
        for det in results:
            if det['score'] >= self.conf_thres:
                x1, y1, x2, y2 = det['box']
                score = det['score']
                class_id = det['class_id']
                boxes.append([x1, y1, x2, y2, score, class_id])
        return boxes

    def get_best_ship_bbox(self, image):
        boxes = self.infer(image)
        if not boxes:
            return None, None
        # Find the best ship box (assume class_id 0 is ship, or use class_name if available)
        best_box = None
        best_score = 0
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            # You may need to adjust class_id or check class_name for 'ship'
            if score > best_score:
                best_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                best_score = score
        return best_box, best_score 