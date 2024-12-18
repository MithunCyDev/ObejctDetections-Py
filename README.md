# Object Detection Using YOLO with OpenCV

This project demonstrates real-time object detection using the YOLO (You Only Look Once) framework with OpenCV. It includes the ability to process frames from a camera feed, identify objects, and display bounding boxes with class labels and confidence scores.

## Features
- Real-time object detection using YOLOv4.
- GPU acceleration if CUDA is available.
- Non-Maximum Suppression (NMS) for removing duplicate detections.
- Saves detected object data to a text file.

## Requirements

- Python 3.6+
- OpenCV 4.5+
- YOLOv4 pre-trained weights and configuration files (`yolov4.weights` and `yolov4.cfg`)
- `objects.txt` containing class names
- A webcam or video feed

## Installation

1. Clone the repository and navigate to the project directory.
2. Install required Python packages:
   ```bash
   pip install opencv-python numpy
   ```
3. Download YOLOv4 pre-trained weights and configuration files from [YOLO official site](https://pjreddie.com/darknet/yolo/).
4. Create an `objects.txt` file containing the list of object class names.

## Usage
1. Run the script:
   ```bash
   python object_detection.py
   ```
2. Detected objects will appear in the video feed with bounding boxes and labels.
3. Detection results will be appended to `detections_log.txt`.

## Sample Output
```
Car,0.85,120,200,50,30
Person,0.78,300,400,60,120
Bicycle,0.67,500,220,80,40
```

## Notes
- Ensure `yolov4.weights`, `yolov4.cfg`, and `objects.txt` are in the working directory.
- Adjust confidence thresholds as needed.
- Press `q` to exit the application.

## License
This project is open-source and can be used freely with attribution.
