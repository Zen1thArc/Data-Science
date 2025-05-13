# Python Fundamentals

### Python Basics

Python adalah bahasa pemrograman yang paling populer di bidang data science karena sintaksnya yang mudah dibaca dan banyaknya library untuk analisis data dan machine learning.

**Variabel dan Tipe Data**

```python
# Variabel dan tipe data dasar
x = 10                # integer
y = 3.14              # float
name = "Data Science" # string
is_valid = True       # boolean

# List (array yang dapat diubah)
numbers = [1, 2, 3, 4, 5]
mixed = [1, "dua", 3.0, True]
numbers.append(6)     # Menambah elemen
numbers[0] = 10       # Mengubah elemen

# Tuples (array yang tidak dapat diubah)
coordinates = (10.5, 20.8)

# Dictionary (key-value pairs)
person = {
    "name": "John",
    "age": 30,
    "skills": ["Python", "SQL"]
}
```

**Kontrol Alur Program**

```python
# If-else statement
x = 10
if x > 5:
    print("x lebih besar dari 5")
elif x == 5:
    print("x sama dengan 5")
else:
    print("x kurang dari 5")

# Loops
# For loop
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

**Fungsi**

```python
# Definisi fungsi
def square(x):
    return x * x

# Fungsi dengan multiple parameters
def calculate_statistics(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    variance = sum((x - average) ** 2 for x in numbers) / len(numbers)
    std_dev = variance ** 0.5
    return average, variance, std_dev

# Memanggil fungsi
nums = [1, 2, 3, 4, 5]
avg, var, std = calculate_statistics(nums)
print(f"Average: {avg}, Variance: {var}, Standard Deviation: {std}")
```

**List Comprehensions**

```python
# List comprehension - cara singkat membuat list
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

**Import Modules**

```python
# Import modul standar
import math
import random
from datetime import datetime

# Import modul data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Pandas Basics

Pandas adalah library Python yang sangat powerful untuk manipulasi dan analisis data.

### Data Manipulation with Pandas

**Series dan DataFrame**

```python
import pandas as pd
import numpy as np

# Membuat Series (1D array dengan label)
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# Membuat DataFrame (2D table)
data = {
    'nama': ['Andi', 'Budi', 'Cindy', 'Deni'],
    'usia': [25, 30, 35, 40],
    'kota': ['Jakarta', 'Bandung', 'Surabaya', 'Yogyakarta']
}
df = pd.DataFrame(data)
print(df)
```

**Membaca dan Menulis Data**

```python
# Membaca dari CSV
df = pd.read_csv('data.csv')

# Membaca dari Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Menulis ke CSV
df.to_csv('output.csv', index=False)

# Menulis ke Excel
df.to_excel('output.xlsx', index=False)
```

**Inspeksi Data**

```python
# Melihat beberapa baris pertama
print(df.head())

# Melihat beberapa baris terakhir
print(df.tail())

# Informasi tentang DataFrame
print(df.info())

# Statistik deskriptif
print(df.describe())

# Dimensi DataFrame
print(df.shape)  # (rows, columns)
```

**Seleksi dan Filtering Data**

```python
# Seleksi kolom
names = df['nama']
subset = df[['nama', 'usia']]

# Seleksi baris dengan loc (label-based)
row = df.loc[0]  # baris pertama
subset = df.loc[0:2]  # baris 0, 1, dan 2

# Seleksi baris dengan iloc (integer-based)
row = df.iloc[0]  # baris pertama
subset = df.iloc[0:2]  # baris 0 dan 1

# Filtering dengan kondisi
adults = df[df['usia'] > 30]
jakarta_people = df[df['kota'] == 'Jakarta']

# Filtering dengan multiple kondisi
jakarta_adults = df[(df['kota'] == 'Jakarta') & (df['usia'] > 30)]
```

**Manipulasi Data**

```python
# Menambah kolom baru
df['kategori_usia'] = ['Muda' if age < 30 else 'Dewasa' for age in df['usia']]

# Mengganti nilai
df['kota'] = df['kota'].replace('Jakarta', 'DKI Jakarta')

# Menghapus kolom
df_subset = df.drop('kategori_usia', axis=1)

# Menghapus baris
df_cleaned = df.drop([0, 2], axis=0)  # Menghapus baris 0 dan 2

# Sorting
df_sorted = df.sort_values('usia', ascending=False)
```

**Grouping dan Aggregasi**

```python
# Group by dan aggregasi
by_city = df.groupby('kota')
city_stats = by_city.agg({
    'usia': ['mean', 'min', 'max', 'count']
})

# Pivot tables
pivot = pd.pivot_table(df,
                       values='usia',
                       index='kota',
                       columns='kategori_usia',
                       aggfunc='mean')
```

**Handling Missing Values**

```python
# Mendeteksi missing values
print(df.isnull().sum())

# Mengisi missing values
df['usia'].fillna(df['usia'].mean(), inplace=True)  # dengan mean
df['kota'].fillna('Unknown', inplace=True)  # dengan value tertentu

# Menghapus baris dengan missing values
df.dropna(inplace=True)
```

**Merge dan Join Data**

```python
# Data tambahan
data2 = {
    'nama': ['Andi', 'Budi', 'Cindy', 'Edi'],
    'gaji': [5000000, 6000000, 7000000, 5500000]
}
df2 = pd.DataFrame(data2)

# Merge (SQL-like join)
merged = pd.merge(df, df2, on='nama', how='inner')  # inner join
merged_left = pd.merge(df, df2, on='nama', how='left')  # left join
merged_outer = pd.merge(df, df2, on='nama', how='outer')  # outer join
```

**Transformasi Data**

```python
# Apply function ke kolom
df['usia_5tahun'] = df['usia'].apply(lambda x: x + 5)

# Apply function ke setiap baris
def get_status(row):
    if row['usia'] < 30:
        return 'Junior'
    elif row['usia'] < 40:
        return 'Mid-level'
    else:
        return 'Senior'

df['status'] = df.apply(get_status, axis=1)
```

## SQL Basics

SQL (Structured Query Language) adalah bahasa standar untuk mengakses dan memanipulasi database.

**Basic SELECT Statement**

```sql
-- Mengambil semua kolom dari tabel
SELECT * FROM employees;

-- Mengambil kolom tertentu
SELECT employee_id, first_name, last_name FROM employees;

-- Menampilkan data unik
SELECT DISTINCT department_id FROM employees;

-- Membatasi jumlah hasil
SELECT * FROM employees LIMIT 10;
```

**WHERE Clause**

```sql
-- Filter data
SELECT * FROM employees WHERE salary > 5000;

-- Multiple conditions
SELECT * FROM employees
WHERE department_id = 10 AND salary > 5000;

SELECT * FROM employees
WHERE department_id = 10 OR department_id = 20;

-- IN operator
SELECT * FROM employees
WHERE department_id IN (10, 20, 30);

-- BETWEEN operator
SELECT * FROM employees
WHERE salary BETWEEN 5000 AND 10000;

-- LIKE operator (pattern matching)
SELECT * FROM employees
WHERE last_name LIKE 'S%';  -- Nama belakang yang dimulai dengan S
```

**ORDER BY**

```sql
-- Sorting data
SELECT * FROM employees ORDER BY salary DESC;  -- Descending
SELECT * FROM employees ORDER BY department_id ASC, salary DESC;  -- Multiple columns
```

**Aggregasi**

```sql
-- Fungsi agregasi
SELECT
    COUNT(*) as total_employees,
    AVG(salary) as average_salary,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary,
    SUM(salary) as total_salary
FROM employees;

-- Dengan Group By
SELECT
    department_id,
    COUNT(*) as employee_count,
    AVG(salary) as average_salary
FROM employees
GROUP BY department_id;

-- Dengan filter setelah group by
SELECT
    department_id,
    COUNT(*) as employee_count,
    AVG(salary) as average_salary
FROM employees
GROUP BY department_id
HAVING COUNT(*) > 5;
```

**JOIN**

```sql
-- Inner Join
SELECT e.employee_id, e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- Left Join
SELECT e.employee_id, e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Right Join
SELECT e.employee_id, e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;

-- Full Outer Join
SELECT e.employee_id, e.first_name, e.last_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.department_id;
```

**Subqueries**

```sql
-- Subquery in WHERE
SELECT employee_id, first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Subquery in FROM
SELECT dept.department_id, dept.employee_count
FROM (
    SELECT department_id, COUNT(*) as employee_count
    FROM employees
    GROUP BY department_id
) dept
WHERE dept.employee_count > 10;
```

**INSERT, UPDATE, DELETE**

```sql
-- Insert data
INSERT INTO employees (employee_id, first_name, last_name, salary)
VALUES (1001, 'John', 'Doe', 5000);

-- Update data
UPDATE employees
SET salary = salary * 1.1
WHERE department_id = 10;

-- Delete data
DELETE FROM employees
WHERE employee_id = 1001;
```

**Creating Tables**

```sql
-- Create table
CREATE TABLE projects (
    project_id INT PRIMARY KEY,
    project_name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    budget DECIMAL(15,2),
    manager_id INT,
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
);
```
