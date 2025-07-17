import torch
from ultralytics import YOLO
from typing import Optional
import logging
import yaml

logger = logging.getLogger(__name__)

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

class YoloV8ShipModel:
    """
    Wrapper for YOLOv8 model for ship detection and size classification.
    Provides training, evaluation, and export utilities.
    """
    def __init__(self, model_cfg: Optional[str] = None, num_classes: int = 5, pretrained: bool = True):
        self.model_cfg = model_cfg or CONFIG['model']['name']
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = YOLO(self.model_cfg)
        logger.info(f"YOLOv8 model initialized with {self.num_classes} classes.")

    def train(self, data_yaml: str, **kwargs):
        """Train the model using the provided data.yaml and hyperparameters."""
        logger.info("Starting training...")
        self.model.train(data=data_yaml, **kwargs)

    def val(self, data_yaml: str, **kwargs):
        """Validate the model using the provided data.yaml."""
        logger.info("Starting validation...")
        return self.model.val(data=data_yaml, **kwargs)

    def predict(self, source: str, **kwargs):
        """Run inference on images or video."""
        logger.info(f"Running inference on {source}")
        return self.model.predict(source=source, **kwargs)

    def export(self, format: str = 'onnx', dynamic: bool = True, simplify: bool = True, **kwargs):
        """Export the model to ONNX or TensorRT format."""
        logger.info(f"Exporting model to {format.upper()} format...")
        return self.model.export(format=format, dynamic=dynamic, simplify=simplify, **kwargs)

    def load_weights(self, weights_path: str):
        """Load model weights from a checkpoint."""
        logger.info(f"Loading weights from {weights_path}")
        self.model = YOLO(weights_path) 