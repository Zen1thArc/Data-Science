# Mathematics and Statistics

# 📐 Calculus Basics

**Calculus** digunakan dalam machine learning untuk optimasi, terutama saat menghitung turunan untuk gradient descent.

---

## 📌 Konsep Utama

-   **Fungsi**: Hubungan antara input dan output.
-   **Turunan (Derivative)**: Tingkat perubahan fungsi.
-   **Gradien**: Vektor turunan pada fungsi multi-variable.
-   **Partial Derivative**: Turunan terhadap satu variabel saat variabel lain dianggap konstan.

---

## 🧠 Gradient Descent

Algoritma optimasi berbasis kalkulus untuk meminimalkan loss function.

### Rumus:

θ = θ - α \* ∇J(θ)

-   `θ`: parameter/model
-   `α`: learning rate
-   `∇J(θ)`: turunan dari fungsi loss

---

## 📌 Contoh Python

```python
import numpy as np

# Fungsi sederhana f(x) = x^2
def f(x): return x**2
def df(x): return 2*x  # turunannya

x = 10
learning_rate = 0.1

for _ in range(10):
    grad = df(x)
    x = x - learning_rate * grad
    print(x)
```

---

# 📊 Descriptive Statistics

Digunakan untuk merangkum dan mendeskripsikan karakteristik utama dari kumpulan data.

---

## 📌 Ukuran Pemusatan

-   **Mean (Rata-rata)**
-   **Median (Tengah)**
-   **Mode (Modus)**

## 📌 Ukuran Sebaran

-   **Range**
-   **Variance**
-   **Standard Deviation**
-   **IQR (Interquartile Range)**

---

## 📌 Contoh Python

```python
import numpy as np
from scipy import stats

data = [10, 12, 14, 12, 18, 20, 22]

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", stats.mode(data, keepdims=False).mode)
print("Std Dev:", np.std(data))
```

---

# 🧪 Hypothesis Testing

Metode statistik untuk menguji asumsi tentang populasi berdasarkan sampel.

---

## 📌 Komponen Utama

-   **H0 (Hipotesis nol)**: Tidak ada efek/perbedaan
-   **H1 (Hipotesis alternatif)**: Ada efek/perbedaan
-   **p-value**: Probabilitas data terjadi jika H0 benar
-   **α (alpha)**: Ambang batas signifikansi (misal: 0.05)

---

## 🧠 Jenis Uji Umum

-   **Z-Test** dan **T-Test** (uji perbedaan rata-rata)
-   **Chi-Square Test** (uji distribusi)
-   **ANOVA** (uji rata-rata banyak grup)

---

## 📌 Contoh Python: Uji T Dua Sampel

```python
from scipy.stats import ttest_ind

group1 = [100, 102, 98, 105, 110]
group2 = [95, 97, 93, 99, 94]

stat, p = ttest_ind(group1, group2)

print("p-value:", p)
if p < 0.05:
    print("Tolak H0 (ada perbedaan signifikan)")
else:
    print("Gagal tolak H0")
```

---

# 🔢 Linear Algebra for Data Science

Dasar penting untuk pemrosesan data, transformasi, dan deep learning.

---

## 📌 Konsep Kunci

-   **Vektor**: Deret angka (1D)
-   **Matriks**: Array 2D
-   **Transpose (Aᵗ)**: Menukar baris ↔ kolom
-   **Dot Product**: Perkalian vektor
-   **Matrix Multiplication**: Kombinasi data/fitur
-   **Eigenvalue dan Eigenvector**

---

## 📌 Contoh Python

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# Perkalian matriks
print("A * B:\n", A @ B)

# Transpose
print("Transpose A:\n", A.T)

# Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
```

---

# 🎲 Probability Basics

Probabilitas adalah pondasi statistik dan model prediktif.

---

## 📌 Aturan Dasar

-   **P(A)**: Probabilitas dari kejadian A
-   **P(A ∩ B)**: A dan B terjadi (AND)
-   **P(A ∪ B)**: A atau B terjadi (OR)
-   **P(A|B)**: Probabilitas A terjadi jika B terjadi (Conditional)

---

## 🧠 Teorema Bayes

P(A|B) = [P(B|A) * P(A)] / P(B)

---

## 📌 Contoh Python

```python
# Probabilitas sederhana
P_A = 0.3
P_B = 0.5
P_A_given_B = 0.6

# Bayes
P_B_given_A = (P_A_given_B * P_B) / P_A
print("P(B|A):", P_B_given_A)
```

---

# 📈 Probability Distributions

Distribusi probabilitas menggambarkan sebaran nilai dari variabel acak.

---

## 📌 Jenis Distribusi

### 🔹 Diskrit

-   **Binomial**: Jumlah sukses dari n percobaan
-   **Poisson**: Jumlah kejadian dalam interval waktu

### 🔸 Kontinu

-   **Normal (Gaussian)**: Bell curve
-   **Uniform**: Semua nilai punya peluang sama
-   **Exponential**: Untuk waktu tunggu antar kejadian

---

## 📌 Contoh Python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Distribusi Normal
x = np.linspace(-4, 4, 100)
y = norm.pdf(x, loc=0, scale=1)

plt.plot(x, y)
plt.title("Normal Distribution")
plt.show()
```
