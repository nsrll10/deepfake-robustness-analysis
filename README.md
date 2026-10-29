# Robustness Analysis of MobileNetV2 and ResNet50 Transfer Learning Models Against JPEG Compression in Deepfake Face Classification

This repository contains the complete experimental code, documentation, and evaluation results from my bachelor's thesis project. This study investigates how image compression algorithms (commonly occurring during social media uploads) impact the performance and robustness of deepfake detection models.

---

## 📌 Project Overview
The proliferation of facial image manipulation (deepfakes) poses serious threats to digital information integrity. While various detection models achieve high accuracy under ideal laboratory conditions, most suffer drastic performance drops when faced with compressed images. 

This project evaluates and compares the robustness of a lightweight Convolutional Neural Network architecture (**MobileNetV2**) versus a deep architecture (**ResNet50**) under simulated JPEG compression stress-testing.

---

## ⚙️ Key Methodologies & Workflow
1. **Dataset:** Utilized a balanced subset of the **Celeb-DF v2** dataset (pre-cropped faces) scaled to a 1:1 ratio (40,288 real images and 40,288 fake images) using **Random Undersampling** to eliminate majority class bias.
2. **Preprocessing & Resizing:** Standardized image dimensions to $224 \times 224$ pixels to meet input requirements for both transfer learning models.
3. **Compression Stress Testing:** Evaluated model performance across four JPEG Quality Factor (QF) scenarios to mimic real-world distribution degradation:
   * **QF 100:** Ideal Baseline (No compression)
   * **QF 90:** Light Compression
   * **QF 60:** Medium Compression (Standard social media distribution)
   * **QF 30:** Extreme Compression (High pixel distortion and blocking artifacts)
4. **Evaluation Metrics:** Analyzed performance using a Confusion Matrix, Accuracy, Precision, Recall, and F1-score.
5. **System Deployment (PoC):** Developed a functional prototype web application using **Streamlit** for real-time deepfake classification.

---

## 📊 Key Findings & Results
* **MobileNetV2 Superiority:** Achieved an accuracy of **83%** under ideal conditions (QF 100) and demonstrated exceptional robustness by retaining **81% accuracy** even under extreme compression (QF 30)—a minimal drop of only 2%.
* **ResNet50 Limitations:** Remained stagnant at **70% accuracy** across all compression levels and exhibited high False Positive rates due to *over-smoothing* when processing degraded pixels.
* **Conclusion:** **MobileNetV2** is proven to be significantly more stable, efficient, and reliable for real-world deployment in deepfake detection systems.

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Environment:** Google Colab
* **Deep Learning Frameworks:** TensorFlow / Keras (MobileNetV2, ResNet50)
* **Data Processing & Computer Vision:** Pandas, NumPy, OpenCV
* **UI Prototyping:** Streamlit

---

## 📁 Repository Structure
```text
├── dataset/            # Information & preprocessing scripts
├── models/             # Trained model checkpoints (.keras)
├── notebooks/          # Jupyter Notebooks for training & evaluation
├── reports/            # Official Thesis PDF Document
└── app.py              # Streamlit web application prototype
