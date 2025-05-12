# Data Scientist

## 🚀 Tahapan Belajar Menjadi Data Scientist

### 1. Dasar-Dasar Matematika dan Statistika

Ilmu Data sangat bergantung pada pemahaman dasar matematika dan statistika:

-   **Statistika Deskriptif & Inferensial**: Digunakan untuk memahami data dan menarik kesimpulan.
-   **Aljabar Linier**: Vektor, matriks, dan operasi matriks sering digunakan dalam algoritma machine learning.
-   **Kalkulus Dasar**: Turunan dan integral digunakan dalam optimasi model.
-   **Probabilitas**: Untuk memahami ketidakpastian dan distribusi data.

#### 📌 Latihan:

1. Hitung mean, median, varians, dan standar deviasi dari dataset `np.random.randint(1, 100, size=20)`.
2. Buat grafik distribusi normal dengan `numpy` dan `matplotlib`.

### 2. Bahasa Pemrograman (Python)

Bahasa yang paling umum digunakan dalam data science. Fokus pada:

#### a. Dasar Python

-   Variabel, tipe data, fungsi, loops, list, dictionary
-   Penanganan file dan input/output dasar

#### 📌 Latihan:

1. Buat program Python yang membaca file CSV, menghitung jumlah baris, dan mean dari kolom tertentu.

#### b. Library Python Penting

##### 1. NumPy

```python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr.mean())
```

##### 2. Pandas

```python
import pandas as pd
data = {'nama': ['Ali', 'Budi'], 'nilai': [80, 90]}
df = pd.DataFrame(data)
print(df['nilai'].mean())
```

##### 3. Matplotlib & Seaborn

```python
import seaborn as sns
sns.histplot(df['nilai'])
```

##### 4. Scikit-learn

```python
from sklearn.linear_model import LinearRegression
X = [[1], [2], [3]]
y = [2, 4, 6]
model = LinearRegression().fit(X, y)
print(model.predict([[4]]))
```

##### 5. Jupyter Notebook

-   Gunakan `%matplotlib inline`, `Markdown`, dan `code cells`.

#### 📌 Latihan:

1. Visualisasikan distribusi nilai menggunakan histogram dan boxplot.
2. Buat regresi linier sederhana dari data buatan.

### 3. Eksplorasi dan Pembersihan Data (EDA)

-   **Drop NA, Fill NA**
-   **Normalisasi & Standarisasi**
-   **Outlier detection** (Z-score, IQR)

#### 📌 Latihan:

1. Bersihkan dataset `titanic.csv` dan tampilkan distribusi umur.
2. Hitung jumlah penumpang laki-laki dan perempuan.

### 4. Machine Learning Dasar

#### 📌 Latihan:

1. Bangun model regresi linier dari dataset `sklearn.datasets.fetch_california_housing()`.
2. Gunakan KNN untuk mengklasifikasikan data iris.
3. Evaluasi dengan confusion matrix dan akurasi.

---

## 🤖 Modeling dalam Data Science

Setelah EDA, tahap penting selanjutnya adalah **modeling** — yaitu membangun model prediktif atau klasifikasi berdasarkan data.

### Jenis-Jenis Modeling:

-   **Regresi**: Untuk prediksi nilai numerik (contoh: harga rumah).
-   **Klasifikasi**: Untuk memetakan data ke dalam kelas (contoh: spam atau bukan).
-   **Clustering**: Untuk mengelompokkan data tanpa label (contoh: segmentasi pelanggan).
-   **Time Series**: Prediksi berdasarkan urutan waktu (contoh: ramalan cuaca).
-   **Rekomendasi**: Sistem saran berbasis perilaku pengguna (contoh: YouTube, Netflix).

### 📌 Latihan:

1. Gunakan `LinearRegression` untuk memprediksi harga rumah.
2. Bangun model klasifikasi menggunakan `LogisticRegression` dan tampilkan confusion matrix.
3. Lakukan clustering dengan `KMeans` pada dataset tanpa label.
4. Prediksi deret waktu menggunakan `ARIMA` atau `Prophet`.

---

### 5. Proyek Mini

#### 📌 Studi Kasus:

-   **Spam Detection**: Dataset SMS Spam
-   **Prediksi Diabetes**: Dataset Pima Indians Diabetes

Langkah:

1. Load data
2. Preprocessing
3. Model training dan evaluasi
4. Visualisasi hasil

### 6. SQL dan Basis Data

#### 📌 Latihan:

1. Buat query untuk menghitung rata-rata, median, dan total data penjualan mingguan.
2. Join dua tabel: pelanggan dan transaksi.

### 7. Tools Pendukung

#### 📌 Latihan:

1. Push proyek mini ke GitHub.
2. Deploy model prediksi ke Streamlit dan jalankan di localhost.

### 8. Lanjut: Deep Learning

#### 📌 Latihan:

1. Buat model MLP untuk prediksi harga rumah.
2. Latih CNN sederhana untuk klasifikasi MNIST dengan TensorFlow.

---

## 📊 Konsep Statistik Dasar untuk Data Science

### 1. Mean, Median, Modus

```python
from scipy import stats
nilai = [70, 80, 90, 60, 100, 80]
print("Mean:", np.mean(nilai))
print("Median:", np.median(nilai))
print("Modus:", stats.mode(nilai, keepdims=True)[0][0])
```

### 2. Regresi

```python
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.show()
```

### 3. Hipotesis Statistik

Uji perbedaan rata-rata:

```python
group1 = [170, 168, 174, 171]
group2 = [165, 160, 163, 167]
from scipy.stats import ttest_ind
t_stat, p = ttest_ind(group1, group2)
```

### 4. Korelasi

```python
from scipy.stats import pearsonr
r, p = pearsonr([1, 2, 3], [2, 4, 6])
```

### 5. Varians dan Standar Deviasi

```python
x = [10, 20, 30]
print(np.var(x), np.std(x))
```

---

## 🧪 Soal Latihan Lengkap

1. Apa arti p-value dalam pengujian hipotesis?
2. Apa perbedaan regresi linier dan logistik?
3. Buat model klasifikasi menggunakan Decision Tree dan evaluasi F1-score.
4. Gunakan Pandas untuk menghitung nilai rata-rata berdasarkan kategori kolom lain.
5. Buat plot hubungan dua variabel numerik menggunakan Seaborn (`sns.scatterplot`).
6. Lakukan feature scaling menggunakan StandardScaler dan MinMaxScaler.
