# 🤖 SmartVision AI

SmartVision AI is a Deep Learning and Computer Vision based web application developed using Streamlit, TensorFlow, and YOLOv8.  

The project performs:

- 🖼 Image Classification using CNN Models
- 🎯 Real-Time Object Detection using YOLOv8
- 📊 Model Performance Comparison
- 📹 Live Webcam Detection
- 📁 Dataset Exploration Dashboard

---

# 🚀 Features

## 🖼 Image Classification
Classifies uploaded images into 26 object categories using:

- MobileNetV2
- ResNet50
- EfficientNetB0

### Functionalities
- Upload image interface
- Predictions from all models
- Confidence score comparison
- Best model prediction
- Interactive visualizations

---

## 🎯 Object Detection
Detects multiple objects in images using YOLOv8.

### Functionalities
- Bounding box detection
- Multi-object recognition
- Confidence scores
- Adjustable detection threshold
- Fast inference speed

---

## 📊 Performance Dashboard
Displays:

- Model Accuracy Comparison
- Inference Speed
- Deep Learning Benchmarking
- Visual Performance Metrics

---

## 📹 Live Webcam Detection
Real-time object detection using webcam feed.

### Functionalities
- Live YOLO detection
- Real-time frame processing
- Instant predictions

---

# 🧠 Deep Learning Models Used

| Model | Purpose |
|---|---|
| MobileNetV2 | Lightweight Image Classification |
| ResNet50 | Deep Residual Learning |
| EfficientNetB0 | Efficient High Accuracy Classification |
| YOLOv8 | Real-Time Object Detection |

---

# 📁 Dataset Information

The project uses a custom dataset consisting of:

- 26 Object Classes
- Classification Dataset
- Detection Dataset (YOLO Format)

### Classes

- airplane
- bed
- bench
- bicycle
- bird
- bottle
- bowl
- bus
- cake
- car
- cat
- chair
- couch
- cow
- cup
- dog
- elephant
- horse
- motorcycle
- person
- pizza
- potted plant
- stop sign
- traffic light
- train
- truck

---

# 🛠 Technologies Used

## Frontend
- Streamlit

## Deep Learning
- TensorFlow / Keras
- Ultralytics YOLOv8

## Libraries
- OpenCV
- NumPy
- Pandas
- Pillow
- Matplotlib

---

# 📂 Project Structure

```bash
SmartVisionAI/
│
├── app.py
│
├── models/
│   ├── mobilenet_weights.weights.h5
│   ├── resnet50_weights.weights.h5
│   ├── efficientnet_weights.weights.h5
│   └── best.pt
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── MobileNetV2.ipynb
│   ├── ResNet50.ipynb
│   ├── EfficientNetB0.ipynb
│   └── YOLOv8.ipynb
│
├── dataset/
│
├── requirements.txt
│
└── README.md

# ⚙ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/SmartVisionAI.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd SmartVisionAI
```

---

## 3️⃣ Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Streamlit Application

```bash
streamlit run app.py
```

---

# 🌐 Application URL

After running the above command, Streamlit will automatically open:

```bash
http://localhost:8501
```

---

# 📦 Required Python Version

```bash
Python 3.9+
```

---

# 🧾 Example requirements.txt

```txt
streamlit
tensorflow
ultralytics
opencv-python
numpy
pandas
pillow
matplotlib
streamlit-option-menu
torch
torchvision
```

📊 Expected Outcomes
Model	Expected Accuracy
MobileNetV2	80% - 90%
ResNet50	85% - 95%
EfficientNetB0	85% - 92%
YOLOv8 mAP@0.5	85% - 90%
🧪 Model Evaluation Metrics

The project evaluates models using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Inference Speed
mAP@0.5
🔥 YOLOv8 Detection Features
Real-time Detection
Multiple Object Detection
High-Speed Inference
Bounding Box Visualization
Confidence Scoring
📸 Application Pages
Page	Description
Home	Project Overview
Image Classification	CNN Predictions
Object Detection	YOLOv8 Detection
Performance Dashboard	Metrics Visualization
Live Webcam	Real-time Detection
Dataset Info	Dataset Details
About	Project Information
📈 Future Enhancements
Video Object Detection
Cloud Deployment
User Authentication
Model Optimization
GPU Acceleration
Advanced Analytics Dashboard
👨‍💻 Developer
SmartVision AI Project

Developed using:

Deep Learning
Transfer Learning
Computer Vision
Streamlit Deployment
📜 License

This project is developed for educational and academic purposes.

⭐ Acknowledgements
TensorFlow
Keras
Ultralytics YOLO
Streamlit
OpenCV
COCO Dataset
