# 🚗 Lane Detection Using Semantic Segmentation

This project presents a **lane detection system based on semantic segmentation**, where lane markings are extracted and highlighted from road scenes using a deep learning model followed by structured post-processing.

The pipeline reflects approaches commonly used in **ADAS and autonomous driving perception systems**, separating semantic understanding from lane geometry logic.

---

## ✨ Demo

<p align="center">
  <img src="assets/demo.gif" width="700">
</p>

> The model performs road and lane segmentation and highlights lane markings consistently across frames.

---

## 📌 Project Overview

This project focuses on **multi-class semantic segmentation** for driving scenes, targeting:

- Road region segmentation (drivable area)
- Background segmentation
- Lane marking segmentation
- Lane visualization derived from predicted lane pixels (no classical lane detection or Hough-based methods)

Supported pipelines:
- 🖼️ Image inference
- 🎥 Video inference
- 🔁 Optional model fine-tuning on custom datasets

---

## 🏗 Model Architecture

- U-Net–style encoder–decoder architecture
- ResNet-50 encoder backbone (pre-trained for feature extraction)
- Multi-class semantic segmentation
- Softmax output over 3 classes:
  - `0` → Background
  - `1` → Road
  - `2` → Lane markings
- Trained using `SparseCategoricalCrossentropy` loss

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

## 📈 Training Details

- Loss function: `SparseCategoricalCrossentropy`
- Optimizer: Adam (`learning_rate = 1e-4`)
- Metrics:
  - Sparse Categorical Accuracy
  - Mean IoU (computed from logits)
  - Lane-specific IoU

Multiple experiments were conducted (5+ model variants) before selecting the final model.

### 📊 Final Model Performance (Epoch 20)

| Metric       | Training | Validation |
|-------------|----------|------------|
| Accuracy    | 0.9972   | 0.9954     |
| Lane IoU    | 0.8517   | 0.8542     |
| Mean IoU    | 0.9456   | 0.9401     |
| Loss        | 0.0069   | 0.0146     |

---

## 🖥️ Inference Pipelines

The repository provides ready-to-run pipelines:

- `src/run_image.py` → perform inference on single images
- `src/run_video.py` → perform inference on videos
- `src/lane_postprocessor.py` → post-processing to highlight lanes on segmented pixels
- `src/inference.py` → utility functions shared across pipelines

> The pipelines operate on the trained model directly. No training is required for inference.

---

## 🔁 Fine-Tuning on Custom Data

This project supports **fine-tuning on custom datasets** through the script:

`training/finetune.py`



Fine-tuning is recommended if:
- You want better performance on a specific country, road type, or camera setup
- Your data distribution differs from the original dataset
- You want to extend or rebalance lane / road classes

### 📒 Fine-Tuning Setup

- Dataset loading and preprocessing handled in `training/dataset.py`
- Multi-class mask generation
- Model compilation with pre-loaded weights
- Training with checkpoints saved automatically
- Evaluation using IoU-based metrics

> Users can optionally adjust learning rate, batch size, and number of epochs inside the script.

---

### 🏷️ Ground Truth Format

Segmentation masks are RGB-encoded and converted into class IDs during training:

| Class        | RGB Value      | Class ID |
|--------------|----------------|----------|
| Background   | (0, 0, 0)      | 0        |
| Road         | (128, 0, 0)    | 1        |
| Lane Marking | (0, 128, 0)    | 2        |

During training, masks are converted to a single-channel tensor of class indices.

---

### 📈 Model Selection

Multiple experiments were conducted (5+ model variants) with different:
- Training schedules
- Data augmentation strategies
- Metric combinations

The final model was selected based on **lane IoU stability** and **visual consistency**, not only numerical metrics.

---


### 🔄 Extending the Model

You may fine-tune the model by:
- Re-training all layers
- Freezing the encoder and training only the decoder
- Adjusting class balance or augmentations

The scripts are structured to support iterative experimentation.


---

## 👤 Author

**Yara Elshehawi**  
Computer Vision & Machine Learning Engineer


🌐 [Portfolio](https://yaraeslamm.github.io)

🔗 [LinkedIn](https://www.linkedin.com/in/yara-eslam-877421212/)

