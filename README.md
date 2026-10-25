# Project 3: Pill Detection and Counting

## Objective
Detect visible pills in an image using YOLO object detection and count
the detected pill instances.

## Technology
- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Google Colab with NVIDIA Tesla T4 GPU

## Dataset
Ultralytics Medical Pills dataset:
https://github.com/ultralytics/assets/releases/download/v0.0.0/medical-pills.zip

Dataset used:
- 92 training images
- 23 validation images
- One class: pill

## Model
YOLOv8 nano pretrained detector, fine-tuned for pill detection.
Input image size: 640 x 640.
The best trained checkpoint is models/best.pt.

## Validation counting results
At confidence threshold 0.25:
- Mean absolute counting error: approximately 0.870 pills
- Exact-count rate: 39.13% (9 of 23 images)

At confidence threshold 0.60:
- Mean absolute counting error: approximately 0.435 pills
- Exact-count rate: 65.22% (15 of 23 images)

Threshold 0.60 is a candidate setting based on this validation set.
These results are not a guarantee of performance on new images.

## Local setup
Install dependencies:
pip install ultralytics

Run:
python src/predict.py path/to/image.jpg --conf 0.60

The script prints the detected count and saves an annotated image.

## Limitations
The dataset and validation set are small. Performance can change with
different pills, packaging, lighting, overlapping objects, or camera views.
This is an educational computer-vision prototype, not a medical device.
Do not use it to determine real medication doses or administer medicines.
