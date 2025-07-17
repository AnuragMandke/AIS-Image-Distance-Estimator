# Ship Distance Estimation Pipeline

This project estimates the distance to ships in images using a two-stage approach:
- **Primary:** Uses AIS (Automatic Identification System) data if available.
- **Fallback:** Uses computer vision (YOLOv8) and triangulation if AIS is not available.

## Features
- Batch processing of all images in a folder
- Annotated output images with bounding box and distance label
- Debug output for all distance calculation parameters
- Easily configurable for different camera setups and ship types

## Requirements
- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Setup
1. Place your YOLOv8 model file (e.g., `ship_detector_final_map982.pt`) in the project directory.
2. Place all input images in the `input_Images` folder (create it if it doesn't exist).
3. Ensure the following camera parameters in `distance_calculator.py` match your camera:
   - `focal_length_mm`: Focal length of your camera lens in millimeters
   - `sensor_width_mm`: Sensor width in millimeters
   - `image_width_px`: **Original image width in pixels** (not the YOLOv8 inference size)
   - `known_objects_db['ship']['real_width_m']`: Real-world width of the ship in meters

## Usage

From the project directory, run:

```sh
python distance_calculator.py
```

- All images in `input_Images` will be processed.
- Annotated images will be saved in `output_Images`.
- The script prints debug information and results for each image.

## Output
- Annotated images with bounding box and distance label in `output_Images`.
- Console output with debug values and distance estimates.

## Notes
- For accurate distance estimation, ensure camera parameters and real ship width are correct.
- The pipeline uses YOLOv8 for ship detection. You can swap in any YOLOv8 `.pt` model.
- If you want to use AIS data, update the code to provide `ais_data` to the `calculate_distance` method.

## Example Directory Structure
```
AIS-Image-Distance_Estimator\
  ├── distance_calculator.py
  ├── ship_detector_final_map982.pt
  ├── input_Images\
  │     ├── image1.jpg
  │     ├── image2.jpg
  │     └── ...
  ├── output_Images\
  │     └── ... (annotated results)
  └── README.md
```

## License
This project is for research and educational use. For commercial use, please contact the author. 
