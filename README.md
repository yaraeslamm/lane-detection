# 🚗 Lane Detection Using Semantic Segmentation

This project presents a **lane detection system based on semantic segmentation**, where lane markings are extracted and highlighted from road scenes using a deep learning model followed by structured post-processing.

The pipeline reflects approaches commonly used in **ADAS and autonomous driving perception systems**, separating semantic understanding from lane geometry logic.

---

## ✨ Features

- Multi-class semantic segmentation:
  - Background
  - Road surface
  - Lane markings
- Lane highlighting derived **directly from segmentation output**
- Robust to illumination and scene variations
- Modular and easy to fine-tune on new datasets

---

## 🧠 Approach Overview

1. A semantic segmentation model predicts pixel-wise class labels.
2. The **lane-marking class** is isolated from the segmentation output.
3. Morphological operations and filtering are applied.
4. Lane markings are overlaid on the road region for visualization.

This design mirrors real-world pipelines where perception models feed downstream geometric reasoning.

---

## 🏗 Model Architecture

- U-Net–style convolutional neural network
- Encoder–decoder structure with skip connections
- Softmax output over 3 classes
- Trained using categorical cross-entropy loss

---

## 📊 Dataset

The model was trained on a **custom enhanced dataset**, originally based on:

**Semantic Segmentation Makassar (IDN) Road Dataset**  
(~374 segmented images)

### Dataset Enhancements
To improve generalization, the dataset was expanded using:
- Random rotations
- Horizontal flips
- Color jittering (brightness, contrast)

Final dataset size: **~1,496 images**

📌 The dataset is publicly available:
- [Kaggle](https://www.kaggle.com/datasets/yaraeslam/enhanced-road-segmentation-dataset/data)
- [Hugging Face](https://huggingface.co/datasets/yaraa11/enhanced-road-segmentation-dataset)

> The dataset is not included in this repository due to size constraints.

---

## 🔁 Training & Fine-Tuning

To fine-tune the model on your own dataset:
1. Prepare RGB masks using the same color coding.
2. Update dataset paths in `data.py`.
3. Adjust the number of classes if needed.
4. Retrain using `train.py`.

The code is designed to be easily adaptable to new road environments.

---

## 📈 Results

- Stable convergence during training
- Accurate separation of road and lane markings
- Clean lane highlighting even under challenging conditions

<p align="center">
  <img src="demo/demo.gif" width="700"/>
</p>

---

## 👤 Author

**Yara Elshehawi**  
Digital Media Engineering  
Focus: Computer Vision & AI
