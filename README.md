# 🩺 Pneumonia Detection from Chest X-Rays

This project is a machine learning application that uses **deep learning (CNNs)** to detect pneumonia from chest X-ray images. Built using **PyTorch**, it classifies medical images as either *Normal* or *Pneumonia*, and includes a web interface using **Streamlit** for real-time interaction.

---

## 🧠 Project Overview

- 📈 **Goal:** Detect signs of pneumonia in chest X-ray images
- 🤖 **Model:** Convolutional Neural Network (CNN)
- 🩻 **Dataset:** RSNA Pneumonia Detection Challenge dataset
- 🧪 **Tech Stack:** Python, PyTorch, Pandas, Matplotlib, Streamlit
- 🌐 **Web App:** Built with Streamlit to upload and classify images in real-time

---

## ✨ Features

- 🧠 Trained a CNN on labeled chest X-ray images (normal vs pneumonia)
- 📊 Visualized model accuracy and loss across training epochs
- 🔬 Included Grad-CAM heatmaps to highlight key image regions (optional)
- 🧪 Streamlit web interface for uploading and classifying chest X-ray images
- 📁 Structured and reproducible pipeline: data preprocessing → training → evaluation → deployment

---

## 📂 Dataset

Dataset used: [RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

- Format: JPEG/PNG chest X-ray images
- Labels: `Normal`, `Pneumonia`

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                                 |
|------------------|-----------------------------------------|
| **Python**        | Programming language                    |
| **PyTorch**       | Deep learning framework                 |
| **Pandas**        | Data handling and preprocessing         |
| **Matplotlib**    | Plotting training performance           |
| **Streamlit**     | Web-based UI for model interaction      |

---
# 💾 Pretrained Model

You can download the pretrained model checkpoint (`model.cpkt`) from the following Google Drive link:

https://drive.google.com/file/d/1ZGZc0gZCqyBvaTdMcFgycuX392ZfhEr4/view?usp=drive_link  

Place this file in the main project before running the application.


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shraddharanjan/pneumonia_detection_app.git
cd pneumonia_detection_app
