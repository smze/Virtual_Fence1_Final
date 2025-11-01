Virtual Fence Project
Overview

The Virtual Fence project aims to develop a system that detects and counts people entering a predefined zone within a crowded street video. The system processes the input video, tracks people, and counts those who enter a specified rectangular zone. The output video shows the detected people with bounding boxes, highlights the predefined zone, and provides a real-time counter of people entering the zone.

The project evaluates and compares the performance of three different detection approaches:

YOLOv5 – A powerful real-time object detection model.

VLM (Vision-Language Model) – A language-vision model for enhanced people detection.

MobileNet – A custom lightweight model for real-time object detection, optimized for edge devices like Raspberry Pi.

Features

Person Detection: Detects people in video frames.

Tracking: Tracks individuals across frames in the video.

Zone Monitoring: Defines and monitors a rectangular zone to count people entering it.

Real-Time Counting: Displays a real-time counter showing how many people have entered the zone.

Method Comparison: Compares the performance of YOLOv5, VLM, and MobileNet.

Frameworks and Methods Used

PyTorch: Used for training and deploying deep learning models.

YOLOv5: A state-of-the-art object detection model that detects people in each frame.

VLM (Vision-Language Model): A model that uses both visual and linguistic features for better object detection.

MobileNet: A lightweight, custom model for real-time object detection, designed for edge devices like Raspberry Pi.

Background Subtractor Detector (BG Subtractor): A traditional method for detecting moving objects using background subtraction.

Model Comparison:

The project compares the accuracy and performance of the three models:

YOLOv5: A deep learning model optimized for high accuracy and speed.

VLM: Uses text-based queries to improve detection in complex or low-quality scenes.

MobileNet: A lightweight deep learning model optimized for edge devices (such as Raspberry Pi) and fast processing.

Datasets
1. CrowdHuman Dataset:

The CrowdHuman dataset is used to detect people in crowded scenes. This dataset contains labeled images of people with bounding boxes, making it perfect for training object detection models.

2. Pexels Dataset:

Additional data from Pexels is used to fine-tune the model. These are images and short videos of crowded urban scenes, labeled using LabelImg.

3. Data Preparation:

Labeling Pexels Images: Images from Pexels are manually labeled using LabelImg, where bounding boxes are drawn around the people in the scenes.

Splitting Video Frames: The input video (e.g., input.mp4) is split into individual frames, which are then labeled for training.

Converting to YOLO Format: The labeled annotations are converted into YOLO format, which is required for training models like YOLOv5.

Installation
Prerequisites:

Ensure you have the following installed:

Python 3.x

OpenCV

PyTorch

YOLOv5 dependencies

NumPy

LabelImg (for labeling images)

Setup:

Clone this repository:

git clone https://github.com/username/virtual-fence-project.git
cd virtual-fence-project


Install required libraries:

pip install -r requirements.txt


Ensure you have LabelImg installed for labeling images:

Install LabelImg by following the instructions on its GitHub page: LabelImg
.

Dataset Preparation:

Download the CrowdHuman dataset:
You can download it directly from the Hugging Face dataset or the official link for CrowdHuman annotations.

Label Pexels Images:

Use LabelImg to label images from Pexels (bounding boxes around people).

Convert the labels into YOLO format using the provided Python script.

Prepare Video Frames:

Split the input video into frames using OpenCV.

Label the frames with LabelImg.

Converting to YOLO Format:

Once the frames and images are labeled, convert the XML annotations to YOLO format using a Python script provided in the repository.

Benchmarking
Evaluation:

After training and implementing the models, we will evaluate each model using the following criteria:

Detection Accuracy (mAP): The mean Average Precision (mAP) is calculated for the object detection models to compare their performance.

Counting Accuracy: We will compare the accuracy of counting people entering the defined zone in real-time across all three models.

Performance Metrics:

Real-time Speed: Evaluate the inference speed (frames per second).

Detection and Counting Precision: Compare the accuracy of people detection and zone entry counting across the three methods.

Output

The output video will include:

Bounding Boxes: Around the detected people.

Highlighted Zone: A clearly marked rectangular zone where people are counted.

Real-Time Counter: Displayed in the top-left corner, showing the count of people entering the zone.

The video output is saved in MP4 format, and its resolution matches that of the input video.

Conclusion

The Virtual Fence project evaluates three different approaches for detecting and counting people in a crowded street scene. This project highlights the trade-offs between speed and accuracy when using state-of-the-art models like YOLOv5, VLM, and lightweight models like MobileNet designed for edge devices like Raspberry Pi.

Contributions

YOLOv5: Detection and tracking using the YOLO framework.

VLM: Integration of vision and language models for better detection in challenging environments.

MobileNet: A custom lightweight model for edge devices, optimized for speed and efficiency.

Next Steps

Fine-tune Models: Fine-tune the models with additional custom data.

Deploy on Raspberry Pi: Optimize the models further for deployment on edge devices.

Improve Counting Mechanism: Refine the counting mechanism to handle occlusions and re-entries.

License

This project is licensed under the MIT License - see the LICENSE
 file for details.
