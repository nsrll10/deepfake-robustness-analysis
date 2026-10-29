# Robustness Analysis of MobileNetV2 and ResNet50 Transfer Learning Models Against JPEG Compression in Deepfake Face Classification

This repository contains the complete experimental code, documentation, and evaluation results from my undergraduate thesis project. This study investigates how image compression algorithms (commonly occurring during social media uploads) impact the performance and robustness of deepfake detection models[cite: 8].

---

## 📌 Project Overview
The proliferation of facial image manipulation (deepfakes) poses serious threats to digital information integrity[cite: 8]. While various detection models achieve high accuracy under ideal laboratory conditions, most suffer drastic performance drops when faced with compressed images[cite: 8]. 

This project evaluates and compares the robustness of a lightweight Convolutional Neural Network architecture (**MobileNetV2**) versus a deep architecture (**ResNet50**) under simulated JPEG compression stress-testing[cite: 8].

---

## ⚙️ Key Methodologies & Workflow
1. **Dataset:** Utilized a balanced subset of the **Celeb-DF v2** dataset (pre-cropped faces) scaled to a 1:1 ratio (40,288 real images and 40,288 fake images) using **Random Undersampling** to eliminate majority class bias[cite: 8, 42].
2. **Preprocessing & Resizing:** Standardized image dimensions to $224 \times 224$ pixels to meet input requirements for both transfer learning models[cite: 42].
3. **Compression Stress Testing:** Evaluated model performance across four JPEG Quality Factor (QF) scenarios to mimic real-world distribution degradation[cite: 43]:
   * **QF 100:** Ideal Baseline (No compression)[cite: 44]
   * **QF 90:** Light Compression[cite: 44]
   * **QF 60:** Medium Compression (Standard social media distribution)[cite: 44]
   * **QF 30:** Extreme Compression (High pixel distortion and blocking artifacts)[cite: 44]
4. **Evaluation Metrics:** Analyzed performance using a Confusion Matrix, Accuracy, Precision, Recall, and F1-score[cite: 8, 45].
5. **System Deployment (PoC):** Developed a functional prototype web application using **Streamlit** for real-time deepfake classification[cite: 8, 39].

---

## 📊 Key Findings & Results
* **MobileNetV2 Superiority:** Achieved an accuracy of **83%** under ideal conditions (QF 100) and demonstrated exceptional robustness by retaining **81% accuracy** even under extreme compression (QF 30)—a minimal drop of only 2%[cite: 8, 57].
* **ResNet50 Limitations:** Remained stagnant at **70% accuracy** across all compression levels and exhibited high False Positive rates due to *over-smoothing* when processing degraded pixels[cite: 8, 57].
* **Conclusion:** **MobileNetV2** is proven to be significantly more stable, efficient, and reliable for real-world deployment in deepfake detection systems[cite: 8].

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python[cite: 8]
* **Environment:** Google Colab[cite: 8]
* **Deep Learning Frameworks:** TensorFlow / Keras (MobileNetV2, ResNet50)[cite: 56]
* **Data Processing & Computer Vision:** Pandas, NumPy, OpenCV[cite: 30, 41]
* **UI Prototyping:** Streamlit[cite: 8]

---

## 📁 Repository Structure
```text
├── dataset/            # Information & preprocessing scripts
├── models/             # Trained model checkpoints (.keras)
├── notebooks/          # Jupyter Notebooks for training & evaluation
├── reports/            # Official Thesis PDF Document
└── app.py              # Streamlit web application prototype
