# 🧹 Data Cleaning Basics

**Data Cleaning** adalah langkah penting dalam data science untuk memastikan kualitas data yang digunakan dalam analisis. Proses ini mencakup berbagai teknik seperti penanganan data hilang (_missing values_), deteksi nilai pencilan (_outlier detection_), dan transformasi data (_data transformation_).

---

## 📌 1. Missing Values (Nilai Kosong)

Nilai kosong atau hilang (NaN) dapat menyebabkan kesalahan dalam analisis dan model prediktif. Penanganannya bisa dengan:

-   Menghapus baris/kolom yang mengandung NaN
-   Mengisi nilai dengan:
    -   Rata-rata (`mean`)
    -   Median
    -   Modus
    -   Nilai khusus seperti "Unknown"

### Contoh:

```python
import pandas as pd
import numpy as np

data = {
    'Name': ['Anna', 'Brian', 'Citra', 'Dodi'],
    'Age': [24, np.nan, 30, np.nan],
    'City': ['Jakarta', 'Bandung', np.nan, 'Surabaya']
}

df = pd.DataFrame(data)

# Menampilkan jumlah missing values per kolom
print(df.isnull().sum())

# Imputasi nilai rata-rata untuk kolom 'Age'
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Imputasi 'Unknown' untuk kolom 'City'
df['City'] = df['City'].fillna('Unknown')

print(df)
```

## 📌 2. Outlier Detection (Deteksi Nilai Pencilan)

Outlier adalah nilai yang sangat berbeda dari mayoritas data dan bisa memengaruhi hasil analisis statistik atau model machine learning.

### Metode Umum:

    IQR (Interquartile Range)
    Z-Score

### Contoh: Menghapus Outlier dengan IQR

```python
# Dataset contoh
df = pd.DataFrame({
    'Income': [4000, 4200, 4300, 100000, 4100, 4050]
})

# Hitung Q1 dan Q3
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1

# Tentukan batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data tanpa outlier
df_clean = df[(df['Income'] >= lower_bound) & (df['Income'] <= upper_bound)]

print(df_clean)
```

## 📌 3. Data Transformation (Transformasi Data)

Transformasi data dilakukan untuk meningkatkan kualitas atau distribusi data sebelum dianalisis atau dimodelkan.

### Tujuan:

Menormalkan distribusi
Menskalakan fitur (feature scaling)
Mengubah format data

### Contoh Teknik:

    -   Min-Max Scaling
    -   Standardization (Z-score)
    -   Log Transformation

### Contoh: Scaling dan Log Transformation

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Dataset contoh
df = pd.DataFrame({'Salary': [3000, 5000, 10000, 20000]})

# Min-Max Scaling
scaler = MinMaxScaler()
df['Salary_Scaled'] = scaler.fit_transform(df[['Salary']])

# Log Transformation
df['Salary_Log'] = np.log(df['Salary'])

print(df)
```

## 📚 Referensi

Scikit-learn: https://scikit-learn.org/
Pandas Documentation: https://pandas.pydata.org/
Seaborn: https://seaborn.pydata.org/
