# 🤖 Machine Learning Fundamentals

**Machine Learning (ML)** adalah cabang dari Artificial Intelligence (AI) yang memungkinkan sistem belajar dari data dan membuat prediksi atau keputusan tanpa diprogram secara eksplisit.

---

## 📌 Konsep Utama

-   **Dataset**: Kumpulan data mentah (fitur + label)
-   **Fitur (features)**: Atribut input dari data
-   **Label / Target**: Nilai yang ingin diprediksi
-   **Model**: Algoritma yang dilatih untuk mengenali pola

---

## 🧠 Tipe Machine Learning

1. **Supervised Learning**: Belajar dari data yang diberi label
2. **Unsupervised Learning**: Belajar dari data tanpa label
3. **Reinforcement Learning** _(lanjutan / lanjutan topik)_

---

## 🚀 Contoh Algoritma Populer

| Tipe         | Contoh Algoritma                           |
| ------------ | ------------------------------------------ |
| Supervised   | Linear Regression, Decision Tree, SVM, KNN |
| Unsupervised | K-Means, PCA, DBSCAN                       |

# 📊 Model Evaluation

Evaluasi model adalah langkah penting untuk mengetahui seberapa baik model machine learning bekerja terhadap data baru (unseen data).

---

## 🎯 Tujuan Evaluasi

-   Mencegah overfitting dan underfitting
-   Memilih model terbaik berdasarkan metrik

---

## 🧪 Teknik Evaluasi

1. **Train-Test Split**

    - Memisahkan dataset menjadi data pelatihan dan pengujian
    - Contoh: 80% training, 20% testing

2. **Cross-Validation (K-Fold CV)**
    - Membagi dataset menjadi K bagian dan melatih model sebanyak K kali

---

## 📐 Metrik Evaluasi Umum

### Untuk Klasifikasi:

-   **Accuracy**: Proporsi prediksi yang benar
-   **Precision**: TP / (TP + FP)
-   **Recall**: TP / (TP + FN)
-   **F1-Score**: Harmonik precision dan recall
-   **Confusion Matrix**: Matriks yang menggambarkan hasil klasifikasi

### Untuk Regresi:

-   **MAE (Mean Absolute Error)**
-   **MSE (Mean Squared Error)**
-   **RMSE (Root MSE)**
-   **R² Score**

---

## 📌 Contoh Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, accuracy_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

print(confusion_matrix(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
```

---

# 🧬 Classification Basics

**Classification** adalah jenis supervised learning di mana output-nya berupa kelas (kategori).

---

## 🔍 Contoh Masalah

-   Spam vs Non-spam email
-   Diagnosis penyakit (Positif / Negatif)
-   Klasifikasi jenis bunga (Iris Setosa, Versicolor, Virginica)

---

## 📦 Dataset Contoh

-   Iris
-   Titanic
-   MNIST (digit tulisan tangan)

---

## 🧠 Algoritma Umum

-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Decision Tree
-   Random Forest
-   Support Vector Machine (SVM)

---

## 📌 Contoh: Klasifikasi dengan Logistic Regression

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Akurasi:", model.score(X_test, y_test))
```

---

# 📈 Regression Basics

**Regression** adalah jenis supervised learning di mana output-nya berupa nilai numerik (kontinu).

---

## 🔍 Contoh Masalah

-   Prediksi harga rumah
-   Perkiraan suhu besok
-   Ramalan penjualan

---

## 📦 Dataset Contoh

-   Boston Housing (deprecated, tapi populer untuk belajar)
-   California Housing
-   Tips (pada library seaborn)

---

## 🧠 Algoritma Umum

-   Linear Regression
-   Ridge/Lasso Regression
-   Random Forest Regressor
-   XGBoost Regressor

---

## 📌 Contoh: Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("R² Score:", model.score(X_test, y_test))
```

---

# 🎓 Supervised vs Unsupervised Learning

---

## ✅ Supervised Learning

Belajar dari data yang memiliki label. Model dilatih untuk memetakan input ke output yang diketahui.

### Contoh:

-   Prediksi harga rumah (regresi)
-   Deteksi email spam (klasifikasi)

### Algoritma:

-   Linear Regression
-   Decision Tree
-   SVM
-   Logistic Regression

---

## ❓ Unsupervised Learning

Belajar dari data tanpa label. Model mencari pola tersembunyi.

### Contoh:

-   Segmentasi pelanggan (clustering)
-   Reduksi dimensi (PCA)

### Algoritma:

-   K-Means
-   DBSCAN
-   PCA

---

## 🆚 Perbandingan

| Aspek  | Supervised Learning  | Unsupervised Learning |
| ------ | -------------------- | --------------------- |
| Label  | Ada                  | Tidak ada             |
| Tujuan | Prediksi             | Eksplorasi            |
| Contoh | Klasifikasi, Regresi | Clustering, PCA       |

📖 Referensi Spesifik per Topik

1. 🧠 Supervised vs Unsupervised Learning
   Scikit-Learn - Supervised Learning

Scikit-Learn - Unsupervised Learning

YouTube: StatQuest - Types of Machine Learning

2. 📊 Model Evaluation
   Scikit-Learn - Model Evaluation

Coursera: Evaluation Metrics

Blog: Analytics Vidhya - Evaluation Metrics

3. 🧬 Classification Basics
   Scikit-Learn - Classification User Guide

Kaggle Learn - Intro to Machine Learning

YouTube: StatQuest - Logistic Regression

4. 📈 Regression Basics
   Scikit-Learn - Linear Models

YouTube: StatQuest - Linear Regression

Medium: Regression Metrics

5. 🤖 Machine Learning (Umum)
   Google - Machine Learning Crash Course (Free)
   Kaggle Learn - Intermediate ML
   fast.ai Practical Deep Learning
