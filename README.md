#  Virtual Fence Project

##  Overview

The **Virtual Fence** project aims to detect, track, and count people entering a predefined region within a crowded street scene.
It processes a video input, detects people in real time, and counts how many individuals cross into a user-defined rectangular zone.

The output is a processed **MP4 video** showing:

* Bounding boxes around detected people,
* A highlighted counting zone, and
* A real-time counter displayed on-screen.

Three distinct methods are implemented and benchmarked:

1. **YOLOv5** – A state-of-the-art object detector.
2. **OMNI VLM (Vision-Language Model)** – A multimodal model for contextual person detection.
3. **MobileNet (Custom Lightweight Model)** – A fast, optimized solution for **Raspberry Pi** and other edge devices.

---

## ⚙️ Features

✅ Person detection and tracking across video frames
✅ Zone monitoring and real-time counting
✅ Video output with live counter overlay
✅ Benchmarking and visual comparison between three methods
✅ Optimized MobileNet model for Raspberry Pi

---

##  Frameworks & Tools

* **PyTorch** – for model development and inference
* **YOLOv5** – pretrained model fine-tuned on the combined dataset
* **OMNI VLM** – for language-guided person detection
* **MobileNetV3** – lightweight CNN customized for real-time inference on low-power devices
* **OpenCV** – for video processing and visualization
* **NumPy / Matplotlib** – for analytics and visualization

---

## 📂 Dataset

Two data sources are used for training and evaluation:

1. **CrowdHuman** (Public Dataset)

   * A large-scale dataset of crowded human scenes
   * Source: [https://huggingface.co/datasets/sshao0516/CrowdHuman](https://huggingface.co/datasets/sshao0516/CrowdHuman)

2. **Custom Pexels Dataset**

   * 50–100 manually collected images  from Pexels and the web
   * All images annotated in **YOLO format** using **MakeSense.ai**

Both datasets are combined during fine-tuning for improved generalization.

---

##  Installation

### Prerequisites

Make sure you have:

* Python ≥ 3.8
* PyTorch ≥ 2.0
* OpenCV ≥ 4.5
* NumPy, Pandas, Matplotlib

### Installation Steps

```bash
# Clone this repository
git clone https://github.com/smze/Virtual_Fence1_Final
cd virtual-fence

# Install dependencies
pip install -r requirements.txt
```

---

## 🧾 Dataset Preparation

### 1️⃣ Labeling

Use **MakeSense.ai** or **LabelImg** to annotate people in your custom images or frames.
Save annotations in **YOLO format** (`.txt` files).

### 2️⃣ Combining Datasets

Use the provided `combine_datasets.py` script to merge your custom dataset with CrowdHuman:

```bash
python combine_datasets.py
```

### 3️⃣ Directory Structure

```
/datasets
  ├── crowdhuman/
  ├── pexels_custom/
  └── combined/
```

---

##  Model Training

### YOLOv5 Training

```bash
python train_yolo.py --data data/combined.yaml --epochs 50
```

### MobileNet Training (Custom)

```bash
python train_mobilenet.py --data data/combined --epochs 50
```

### VLM (OMNI) Inference

```bash
python vlm_infer.py --input video.mp4 --zone coordinates.json
```

---

## 🎥 Video Inference & Output

For all three methods, run:

```bash
python main_yolo.py        # YOLOv5 inference
python main_vlm.py         # OMNI-VLM inference
python main_mobilenet.py   # MobileNet inference
```

Each script will:

* Draw bounding boxes around detected people
* Highlight the counting region
* Display the live counter
* Export output as `output_video.mp4`

---

## 📊 Benchmark & Evaluation

Run the benchmarking script to compare all methods:

```bash
python benchmark.py
```

### Metrics Evaluated

* **Detection Accuracy (mAP)** – Average precision for person detection
* **Counting Accuracy (%)** – Correct count ratio compared to manual annotations
* **Inference Speed (FPS)** – Average frames per second during processing

### Example Output Table

| Model     | mAP (%) | Counting Accuracy (%) | FPS (Raspberry Pi) | FPS (Desktop) |
| --------- | ------- | --------------------- | ------------------ | ------------- |
| YOLOv5    | 92.4    | 95.1                  | 10                 | 35            |
| OMNI VLM  | 90.8    | 94.3                  | 8                  | 28            |
| MobileNet | 88.6    | 91.0                  | **18**             | **45**        |

---

##  Notes on Raspberry Pi Optimization

* MobileNet model is quantized and pruned for faster inference.
* OpenCV’s `cv2.dnn` backend and `cv2.VideoWriter` are used for efficiency.
* Model weights are exported as `.tflite` for TensorFlow Lite compatibility.

---

##  Repository Structure

```
/virtual-fence
│
├── datasets/
│   ├── crowdhuman/
│   ├── pexels_custom/
│   └── combined/
│
├── models/
│   ├── yolov5/
│   ├── mobilenet/
│   └── vlm/
│
├── scripts/
│   ├── train_yolo.py
│   ├── train_mobilenet.py
│   ├── vlm_infer.py
│   ├── main_yolo.py
│   ├── main_mobilenet.py
│   ├── main_vlm.py
│   └── benchmark.py
│
├── output/
│   └── output_video.mp4
│
└── README.md
```

---

## 📈 Results Summary

* **YOLOv5** achieved the highest detection precision.
* **OMNI VLM** performed best in crowded or occluded scenes.
* **MobileNet** provided the fastest inference and lowest energy consumption, ideal for Raspberry Pi.

---

## 💬 Conclusion

The **Virtual Fence** system successfully detects, tracks, and counts people entering a defined zone using three different approaches.
Among them, **MobileNet** offers the most practical trade-off between speed and accuracy for real-time edge deployment.

This repository includes complete source code, datasets, output samples, and benchmarking scripts for reproducibility.























