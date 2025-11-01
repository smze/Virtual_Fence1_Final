# Virtual Fence Project

## Overview
The **Virtual Fence** project is designed to detect and count people entering a predefined zone within a crowded street video. The system processes an input video, detects people, tracks them, and counts those who enter the specified rectangular zone. The output video highlights the detected people with bounding boxes and shows the real-time counter of people entering the region.

## Project Goals
The primary goal of this project is to create a robust system for detecting, tracking, and counting people in crowded environments using computer vision techniques. The system is optimized for fast execution, making it suitable for real-time applications and low-resource devices like Raspberry Pi.

## Key Features:
- Person detection and tracking using pre-trained models (YOLO, VLM).
- Real-time counting of people entering a predefined zone.
- Output video with bounding boxes and entry counter.
- Benchmarking of three different approaches (OMNI VLM, YOLO, Custom Method) for performance comparison.

## Libraries Used:
- PyTorch
- OpenCV
- YOLOv5 (pre-trained weights)
- Custom VLM (Visual Language Models)
- NumPy
- Pandas

## Speed Optimization:
Performance optimization for faster inference, especially on low-power devices such as Raspberry Pi.

## Dataset:
The dataset consists of custom-collected videos and images of crowded street scenes, annotated with bounding boxes for people. Additionally, public datasets (such as CrowdHuman) are used for model training.

## Annotation Platform:
The images and videos are annotated using the **MakeSense** platform, with annotations in YOLO format.

## Benchmarking:
Performance is evaluated using metrics such as accuracy and inference time, with detailed comparison across different methods.
