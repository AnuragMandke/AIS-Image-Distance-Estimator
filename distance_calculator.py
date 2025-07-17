import logging
import math
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DistanceCalculator")

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    :param coord1: (lat1, lon1)
    :param coord2: (lat2, lon2)
    :return: distance in meters
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class DistanceCalculator:
    """
    Calculates distance to an object using AIS data or image-based triangulation.
    """

    def __init__(self, camera_params: Dict[str, Any], known_objects_db: Dict[str, Dict[str, Any]]):
        """
        :param camera_params: Dict with keys: 'focal_length_mm', 'sensor_width_mm', 'sensor_height_mm', 'image_width_px', 'image_height_px', 'camera_position' (lat, lon)
        :param known_objects_db: Dict mapping object_type to {'real_width_m': float, 'real_height_m': float}
        """
        self.camera_params = camera_params
        self.known_objects_db = known_objects_db

    def calculate_distance(self, image: np.ndarray, ais_data: Optional[Dict[str, float]] = None, object_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrates distance calculation using AIS or triangulation.
        :param image: Image as numpy array (BGR)
        :param ais_data: Dict with 'lat' and 'lon' keys (optional)
        :param object_type: String key for known object type (optional)
        :return: Dict with 'distance_m', 'confidence', 'method', 'error' (if any)
        """
        try:
            if ais_data and self._validate_ais_data(ais_data):
                logger.info("Attempting AIS-based distance calculation.")
                distance, confidence = self._distance_from_ais(ais_data, self.camera_params['camera_position'])
                return {
                    "distance_m": distance,
                    "confidence": confidence,
                    "method": "AIS",
                    "error": None
                }
            elif object_type:
                logger.info("Falling back to triangulation-based distance calculation.")
                distance, confidence = self._distance_from_triangulation(image, object_type)
                return {
                    "distance_m": distance,
                    "confidence": confidence,
                    "method": "Triangulation",
                    "error": None
                }
            else:
                logger.error("No valid method available for distance calculation.")
                return {
                    "distance_m": None,
                    "confidence": 0.0,
                    "method": None,
                    "error": "No valid method available (missing AIS data and/or object type)."
                }
        except Exception as e:
            logger.exception("Error during distance calculation.")
            return {
                "distance_m": None,
                "confidence": 0.0,
                "method": None,
                "error": str(e)
            }

    def _validate_ais_data(self, ais_data: Dict[str, float]) -> bool:
        """
        Validates AIS data quality.
        """
        try:
            lat, lon = ais_data['lat'], ais_data['lon']
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                logger.warning("AIS data out of bounds.")
                return False
            # Add more checks as needed (e.g., timestamp freshness)
            return True
        except Exception:
            logger.warning("AIS data missing required fields.")
            return False

    def _distance_from_ais(self, ais_data: Dict[str, float], camera_position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculates distance using AIS coordinates and camera position.
        :return: (distance in meters, confidence score)
        """
        obj_coord = (ais_data['lat'], ais_data['lon'])
        cam_coord = camera_position
        distance = haversine(cam_coord, obj_coord)
        # Confidence can be based on AIS data quality, GPS accuracy, etc.
        confidence = 0.95  # Example: high confidence if AIS is available
        logger.info(f"AIS-based distance: {distance:.2f} m")
        return distance, confidence

    def _distance_from_triangulation(self, image: np.ndarray, object_type: str) -> Tuple[Optional[float], float]:
        """
        Calculates distance using triangulation.
        :return: (distance in meters, confidence score)
        """
        if object_type not in self.known_objects_db:
            logger.error(f"Unknown object type: {object_type}")
            return None, 0.0

        real_width = self.known_objects_db[object_type]['real_width_m']
        focal_length_mm = self.camera_params['focal_length_mm']
        sensor_width_mm = self.camera_params['sensor_width_mm']
        image_width_px = self.camera_params['image_width_px']

        # Detect object in image
        bbox = self._detect_object_dimensions(image, object_type)
        if bbox is None:
            logger.error("Object detection failed.")
            return None, 0.0

        x, y, w, h = bbox
        object_width_px = w

        # Triangulation formula
        focal_length_px = (focal_length_mm / sensor_width_mm) * image_width_px
        distance = (real_width * focal_length_px) / object_width_px

        # Debug prints for all relevant values
        print(f"[DEBUG] Focal length (mm): {focal_length_mm}")
        print(f"[DEBUG] Sensor width (mm): {sensor_width_mm}")
        print(f"[DEBUG] Image width (px): {image_width_px}")
        print(f"[DEBUG] Real ship width (m): {real_width}")
        print(f"[DEBUG] Bounding box width (px): {object_width_px}")
        print(f"[DEBUG] Calculated distance (m): {distance}")

        # Confidence can be based on detection quality, image clarity, etc.
        confidence = 0.7  # Example: lower than AIS
        logger.info(f"Triangulation-based distance: {distance:.2f} m")
        return distance, confidence

    def _detect_object_dimensions(self, image: np.ndarray, object_type: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects object and returns bounding box (x, y, w, h) in pixels.
        Uses YOLOv8 with the custom ship detector model.
        """
        logger.info(f"Detecting object of type: {object_type} using YOLOv8 custom model")
        if object_type != 'ship':
            logger.warning("YOLOv8 detection is only set up for 'ship' class.")
            return None
        try:
            model = YOLO('ship_detector_final_map982.pt')
        except Exception as e:
            logger.error(f"Could not load YOLOv8 model: {e}")
            return None
        results = model(image)
        best_box = None
        best_conf = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # If your model is trained for ship only, cls_id will always be 0
                if conf > best_conf:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    best_box = (x1, y1, x2 - x1, y2 - y1)
                    best_conf = conf
        if best_box:
            logger.info(f"YOLOv8 detected bounding box: {best_box} with conf {best_conf}")
            return best_box
        logger.warning("YOLOv8 did not detect the specified object type.")
        return None

# Example usage
if __name__ == "__main__":
    # Example camera parameters and known objects
    camera_params = {
        'focal_length_mm': 35,
        'sensor_width_mm': 36,
        'sensor_height_mm': 24,
        'image_width_px': 4000,  # Set to your original image width
        'image_height_px': 3000, # Set to your original image height
        'camera_position': (37.7749, -122.4194)  # Example: San Francisco
    }
    known_objects_db = {
        'ship': {'real_width_m': 30.0, 'real_height_m': 10.0},
        'buoy': {'real_width_m': 2.0, 'real_height_m': 3.0}
    }

    # Batch processing for all images in input_Images
    input_dir = 'input_Images'
    output_dir = 'output_Images'
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, '*.*'))
    print(f"Found {len(image_paths)} images in {input_dir}")
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image: {img_path}")
            continue
        # Adjust camera params for this image size
        camera_params2 = camera_params.copy()
        camera_params2['image_width_px'] = image.shape[1]
        camera_params2['image_height_px'] = image.shape[0]
        calc2 = DistanceCalculator(camera_params2, known_objects_db)
        result2 = calc2.calculate_distance(image, ais_data=None, object_type='ship')
        print(f"{os.path.basename(img_path)}: {result2}")
        # Annotate and save image if detection succeeded
        bbox = calc2._detect_object_dimensions(image, 'ship')
        if bbox:
            x, y, w, h = bbox
            annotated = image.copy()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Distance: {result2['distance_m']:.2f} m"
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            print(f"Annotated image saved as {out_path}")
    print("Batch processing complete.") 