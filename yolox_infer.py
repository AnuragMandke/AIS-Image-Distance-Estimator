import sys
import os
import cv2
import torch
import numpy as np

# Add YOLOX repo to path
YOLOX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YOLOX')
sys.path.insert(0, YOLOX_PATH)

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import ValTransform

class YOLOXInfer:
    def __init__(self, config_path, weights_path, device="cpu", conf_thres=0.3, nms_thres=0.45):
        self.device = device
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        # Load experiment config
        exp = get_exp(config_path, None)
        self.exp = exp
        self.model = exp.get_model().to(device)
        self.model.eval()
        ckpt = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        self.model = fuse_model(self.model)
        self.preproc = ValTransform(legacy=False)
        self.class_names = exp.class_names if hasattr(exp, "class_names") else None

    def infer(self, image):
        # image: np.ndarray (BGR)
        img_h, img_w = image.shape[:2]
        input_size = self.exp.test_size if hasattr(self.exp, "test_size") else (640, 640)
        img, ratio = self.preproc(image, input_size, swap=(2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.exp.num_classes, self.conf_thres, self.nms_thres, class_agnostic=True)
        if outputs[0] is None:
            return None, None
        # outputs[0]: (num_boxes, 7) [x1, y1, x2, y2, score, class, ...]
        boxes = outputs[0].cpu().numpy()
        # Rescale boxes to original image size
        boxes[:, :4] /= ratio
        return boxes, self.class_names

    def get_best_ship_bbox(self, image):
        boxes, class_names = self.infer(image)
        if boxes is None or len(boxes) == 0:
            return None, None
        # Find the best ship box (assume 'ship' in class_names)
        if class_names and 'ship' in class_names:
            ship_idx = class_names.index('ship')
        else:
            # fallback: use class 0
            ship_idx = 0
        best_box = None
        best_score = 0
        for box in boxes:
            x1, y1, x2, y2, score, cls_id = box[:6]
            if int(cls_id) == ship_idx and score > best_score:
                best_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                best_score = score
        return best_box, best_score 