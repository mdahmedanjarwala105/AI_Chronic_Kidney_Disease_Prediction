# Chronic Kidney Disease (CKD) Prediction using Neural Networks

## 📌 Project Overview
This project predicts **Chronic Kidney Disease (CKD)** from patient medical test data using a **Neural Network (Keras + TensorFlow)**.  
The goal is to classify whether a patient has CKD (`1`) or Not CKD (`0`) based on medical features.

---

## 📊 Dataset
- Source: `kidney_disease.csv`  
- Selected Features:  
  - `sg` (Specific Gravity)  
  - `al` (Albumin)  
  - `sc` (Serum Creatinine)  
  - `hemo` (Hemoglobin)  
  - `pcv` (Packed Cell Volume)  
  - `wc` (White Blood Cell Count)  
  - `rc` (Red Blood Cell Count)  
  - `htn` (Hypertension)  
- Target: `classification` (CKD / Not CKD)  

---

## 🛠 Preprocessing
- Dropped missing values.  
- Encoded categorical features using **LabelEncoder**.  
- Train-test split: 80% training, 20% testing.  
- Features scaled to [0,1] using **MinMaxScaler**.  

Formula:
\[
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
\]

---

## 🧠 Model Architecture
- **Input Layer:** neurons = number of features (8).  
- **Hidden Layer:** Dense(256, activation=ReLU, kernel_initializer="he_normal")  
  - ReLU formula:  
  \[
  f(z) = \max(0, z)
  \]  
- **Output Layer:** Dense(1, activation=Sigmoid)  
  - Sigmoid formula:  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]  

---

## ⚙️ Training
- **Loss Function:** Binary Crossentropy  
  \[
  \text{Loss} = - \big( y \cdot \log(p) + (1-y) \cdot \log(1-p) \big)
  \]  
- **Optimizer:** Adam  
- **Metric:** Accuracy  
- **Callback:** EarlyStopping(patience=10, restore_best_weights=True)  

---

## 📈 Evaluation Metrics
- **Accuracy**  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)

- **Precision**  
  Precision = TP / (TP + FP)

- **Recall (Sensitivity)**  
  Recall = TP / (TP + FN)

- **F1-score**  
  F1 = 2 * (Precision * Recall) / (Precision + Recall)

- **Confusion Matrix**  
     [[ TN  FP ]  
      [ FN  TP ]]
  
---

## 📊 Results
- Training Accuracy: ~99%  
- Validation Accuracy: ~99%  
- Test Accuracy: 100% on given test set.  
- Confusion Matrix: perfect classification (no FP, no FN).  

---

## 📉 Training Graphs
- Accuracy vs Epochs (Train vs Validation)  
- Loss vs Epochs (Train vs Validation)
  
---
