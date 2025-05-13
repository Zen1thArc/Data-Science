# Materi Tambahan untuk Pembelajaran Data Science

# FUNDAMENTAL DATA SCIENCE

## 00_Fundamentals

### Programming_Basics

### Mathematics_and_Statistics

#### Descriptive_Statistics

Statistik deskriptif adalah metode untuk merangkum dan mendeskripsikan data.

**Ukuran Pemusatan**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data sampel
data = [12, 15, 18, 22, 30, 31, 35, 40, 41, 45, 50]

# Mean (rata-rata)
mean = np.mean(data)
print(f"Mean: {mean}")

# Median (nilai tengah)
median = np.median(data)
print(f"Median: {median}")

# Mode (nilai yang paling sering muncul)
from scipy import stats
mode = stats.mode(data)[0][0]
print(f"Mode: {mode}")

# Quartiles (pembagian data menjadi 4 bagian)
q1 = np.percentile(data, 25)  # Quartil pertama
q2 = np.percentile(data, 50)  # Quartil kedua (median)
q3 = np.percentile(data, 75)  # Quartil ketiga
print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")
```

**Ukuran Penyebaran**

```python
# Range (rentang)
data_range = max(data) - min(data)
print(f"Range: {data_range}")

# Variance (varians)
variance = np.var(data, ddof=1)  # ddof=1 untuk sample variance
print(f"Variance: {variance}")

# Standard Deviation (standar deviasi)
std_dev = np.std(data, ddof=1)
print(f"Standard Deviation: {std_dev}")

# Interquartile Range (IQR)
iqr = q3 - q1
print(f"IQR: {iqr}")

# Coefficient of Variation (koefisien variasi)
cv = (std_dev / mean) * 100
print(f"Coefficient of Variation: {cv}%")
```

**Distribusi Data**

```python
# Histogram untuk visualisasi distribusi
plt.figure(figsize=(10, 6))
plt.hist(data, bins=5, edgecolor='black', alpha=0.7)
plt.title('Histogram Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Skewness (kemiringan)
from scipy.stats import skew
skewness = skew(data)
print(f"Skewness: {skewness}")
# Positif: ekor panjang ke kanan
# Negatif: ekor panjang ke kiri
# Mendekati 0: simetris

# Kurtosis (keruncingan)
from scipy.stats import kurtosis
kurt = kurtosis(data)
print(f"Kurtosis: {kurt}")
# Positif: lebih runcing dari distribusi normal
# Negatif: lebih datar dari distribusi normal
# Mendekati 0: seperti distribusi normal
```

**Visualisasi Deskriptif**

```python
# Box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, vert=False)
plt.title('Box Plot Data')
plt.xlabel('Value')
plt.grid(axis='x', alpha=0.75)
plt.show()

# QQ plot untuk memeriksa normalitas
from scipy import stats
plt.figure(figsize=(10, 6))
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.grid(True)
plt.show()
```

**Analisis Deskriptif dengan Pandas**

```python
# Membuat DataFrame untuk analisis
df = pd.DataFrame({'values': data})

# Statistik deskriptif lengkap
desc_stats = df.describe()
print(desc_stats)

# Menambahkan statistik tambahan
desc_stats.loc['skew'] = df.skew().values
desc_stats.loc['kurtosis'] = df.kurtosis().values
desc_stats.loc['median'] = df.median().values
desc_stats.loc['mode'] = df.mode().values[0]
print(desc_stats)
```

#### Probability_Basics

Probabilitas adalah studi tentang kemungkinan kejadian.

**Konsep Dasar**

```python
# Definisi probabilitas: P(A) = jumlah hasil yang diinginkan / jumlah hasil yang mungkin

# Contoh: probabilitas mendapatkan angka genap pada dadu 6 sisi
p_even = 3/6  # angka genap: 2, 4, 6
print(f"P(angka genap) = {p_even}")

# Probabilitas total: P(A or B) = P(A) + P(B) - P(A and B)
# Contoh: probabilitas mendapatkan angka genap ATAU lebih besar dari 4
p_greater_than_4 = 2/6  # 5, 6
p_even_and_greater = 1/6  # hanya 6
p_even_or_greater = p_even + p_greater_than_4 - p_even_and_greater
print(f"P(even OR > 4) = {p_even_or_greater}")

# Probabilitas bersyarat: P(A|B) = P(A and B) / P(B)
# Contoh: probabilitas mendapatkan angka 6 jika diketahui angka yang keluar genap
p_6_given_even = (1/6) / (3/6)
print(f"P(6|even) = {p_6_given_even}")
```

**Kombinasi dan Permutasi**

```python
import math

# Permutasi: nPr = n! / (n-r)!
def permutation(n, r):
    return math.factorial(n) // math.factorial(n-r)

# Kombinasi: nCr = n! / (r! * (n-r)!)
def combination(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n-r))

# Contoh: dari 10 orang, berapa cara memilih 3 orang untuk komite?
ways_to_select_committee = combination(10, 3)
print(f"Ways to select committee: {ways_to_select_committee}")

# Contoh: dari 10 orang, berapa cara mengurutkan 3 orang untuk posisi ketua, wakil, dan sekretaris?
ways_to_select_positions = permutation(10, 3)
print(f"Ways to select positions: {ways_to_select_positions}")

# Dengan library scipy
from scipy.special import comb, perm
print(f"Combination (scipy): {comb(10, 3, exact=True)}")
print(f"Permutation (scipy): {perm(10, 3, exact=True)}")
```

**Simulasi Probabilitas**

```python
import numpy as np

# Simulasi melempar dadu 1000 kali
np.random.seed(42)  # untuk hasil yang reproducible
rolls = np.random.randint(1, 7, size=1000)

# Probabilitas empiris mendapatkan angka genap
empirical_p_even = np.mean(rolls % 2 == 0)
print(f"Empirical P(even): {empirical_p_even}")

# Simulasi melempar koin adil 1000 kali
coin_flips = np.random.choice(['H', 'T'], size=1000)
heads_count = np.sum(coin_flips == 'H')
empirical_p_heads = heads_count / 1000
print(f"Empirical P(heads): {empirical_p_heads}")

# Visualisasi hasil eksperimen
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(rolls, bins=np.arange(1, 8) - 0.5, edgecolor='black', rwidth=0.8)
plt.title('Hasil 1000 Lemparan Dadu')
plt.xlabel('Nilai Dadu')
plt.ylabel('Frekuensi')
plt.xticks(range(1, 7))

plt.subplot(1, 2, 2)
plt.bar(['Heads', 'Tails'], [np.sum(coin_flips == 'H'), np.sum(coin_flips == 'T')],
        color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Hasil 1000 Lemparan Koin')
plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()
```

**Hukum Bayes**

```python
# Teorema Bayes: P(A|B) = P(B|A) * P(A) / P(B)

# Contoh kasus: Tes medis
# Diketahui:
# - 1% populasi menderita penyakit (prevalence)
# - Sensitivitas tes: 95% (true positive rate)
# - Spesifisitas tes: 90% (true negative rate)

# Probabilitas prior
p_disease = 0.01  # P(D)
p_no_disease = 0.99  # P(~D)

# Probabilitas bersyarat
p_positive_if_disease = 0.95  # P(+|D)
p_negative_if_disease = 0.05  # P(-|D)
p_positive_if_no_disease = 0.10  # P(+|~D) = 1 - spesifisitas
p_negative_if_no_disease = 0.90  # P(-|~D)

# Probabilitas positif total
p_positive = p_positive_if_disease * p_disease + p_positive_if_no_disease * p_no_disease

# Probabilitas memiliki penyakit jika hasil tes positif (posterior)
p_disease_if_positive = (p_positive_if_disease * p_disease) / p_positive

print(f"P(penyakit | tes positif) = {p_disease_if_positive:.4f} = {p_disease_if_positive*100:.2f}%")

# Kesimpulan: Meskipun tes positif, probabilitas benar-benar memiliki penyakit hanya sekitar 8.7%
```

#### Probability_Distributions

Distribusi probabilitas menggambarkan pola kemungkinan hasil dari suatu eksperimen acak.

**Distribusi Diskrit**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Distribusi Bernoulli (kejadian biner: sukses/gagal)
p = 0.3  # probabilitas sukses
bern = stats.bernoulli(p)

# PMF (Probability Mass Function)
x = np.array([0, 1])
pmf = bern.pmf(x)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.bar(x, pmf, width=0.4)
plt.title(f'Distribusi Bernoulli (p={p})')
plt.xlabel('Nilai')
plt.ylabel('Probabilitas')
plt.xticks([0, 1], ['Gagal (0)', 'Sukses (1)'])

# 2. Distribusi Binomial (jumlah sukses dalam n percobaan)
n = 10  # jumlah percobaan
p = 0.3  # probabilitas sukses
binom = stats.binom(n, p)

# PMF (Probability Mass Function)
x = np.arange(0, n+1)
pmf = binom.pmf(x)

plt.subplot(2, 2, 2)
plt.bar(x, pmf, width=0.4)
plt.title(f'Distribusi Binomial (n={n}, p={p})')
plt.xlabel('Jumlah Sukses')
plt.ylabel('Probabilitas')

# 3. Distribusi Poisson (jumlah kejadian dalam interval)
mu = 3  # rata-rata jumlah kejadian
poisson = stats.poisson(mu)

# PMF (Probability Mass Function)
x = np.arange(0, 12)
pmf = poisson.pmf(x)

plt.subplot(2, 2, 3)
plt.bar(x, pmf, width=0.4)
plt.title(f'Distribusi Poisson (λ={mu})')
plt.xlabel('Jumlah Kejadian')
plt.ylabel('Probabilitas')

# 4. Distribusi Geometrik (percobaan pertama yang sukses)
p = 0.3  # probabilitas sukses
geom = stats.geom(p)

# PMF (Probability Mass Function)
x = np.arange(1, 12)
pmf = geom.pmf(x)

plt.subplot(2, 2, 4)
plt.bar(x, pmf, width=0.4)
plt.title(f'Distribusi Geometrik (p={p})')
plt.xlabel('Jumlah Percobaan Hingga Sukses')
plt.ylabel('Probabilitas')

plt.tight_layout()
plt.show()

# Simulasi distribusi diskrit
n_samples = 1000

# Bernoulli
bern_samples = bern.rvs(size=n_samples)
print(f"Mean of Bernoulli samples: {np.mean(bern_samples):.4f} (Expected: {p})")

# Binomial
binom_samples = binom.rvs(size=n_samples)
print(f"Mean of Binomial samples: {np.mean(binom_samples):.4f} (Expected: {n*p})")

# Poisson
poisson_samples = poisson.rvs(size=n_samples)
print(f"Mean of Poisson samples: {np.mean(poisson_samples):.4f} (Expected: {mu})")
```

**Distribusi Kontinu**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Distribusi Normal (Gaussian)
mu = 0      # mean
sigma = 1   # standard deviation
x = np.linspace(-4, 4, 1000)
pdf_normal = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(x, pdf_normal)
plt.title(f'Distribusi Normal (μ={mu}, σ={sigma})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Menunjukkan area di bawah kurva (CDF)
for area in [0.68, 0.95, 0.997]:
    z = stats.norm.ppf((1 + area) / 2)
    x_fill = np.linspace(-z, z, 1000)
    plt.fill_between(x_fill, stats.norm.pdf(x_fill, mu, sigma), alpha=0.2)
    plt.axvline(z, color='red', linestyle='--', alpha=0.3)
    plt.axvline(-z, color='red', linestyle='--', alpha=0.3)

# 2. Distribusi Uniform
a, b = 0, 1  # lower and upper bounds
x = np.linspace(-0.5, 1.5, 1000)
pdf_uniform = stats.uniform.pdf(x, a, b-a)

plt.subplot(2, 2, 2)
plt.plot(x, pdf_uniform)
plt.title(f'Distribusi Uniform (a={a}, b={b})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# 3. Distribusi Eksponensial
lambda_exp = 0.5  # rate parameter
x = np.linspace(0, 10, 1000)
pdf_exp = stats.expon.pdf(x, scale=1/lambda_exp)

plt.subplot(2, 2, 3)
plt.plot(x, pdf_exp)
plt.title(f'Distribusi Eksponensial (λ={lambda_exp})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# 4. Distribusi t-Student
df = 3  # degrees of freedom
x = np.linspace(-4, 4, 1000)
pdf_t = stats.t.pdf(x, df)
pdf_normal = stats.norm.pdf(x, 0, 1)  # for comparison

plt.subplot(2, 2, 4)
plt.plot(x, pdf_t, label=f't-distribution (df={df})')
plt.plot(x, pdf_normal, 'r--', label='Normal distribution')
plt.title('Distribusi t vs Normal')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Simulasi dan sampling
n_samples = 10000

# Normal
normal_samples = stats.norm.rvs(mu, sigma, size=n_samples)
print(f"Mean of Normal samples: {np.mean(normal_samples):.4f} (Expected: {mu})")
print(f"Std of Normal samples: {np.std(normal_samples):.4f} (Expected: {sigma})")

# Uniform
uniform_samples = stats.uniform.rvs(a, b-a, size=n_samples)
print(f"Mean of Uniform samples: {np.mean(uniform_samples):.4f} (Expected: {(a+b)/2})")

# Eksponensial
exp_samples = stats.expon.rvs(scale=1/lambda_exp, size=n_samples)
print(f"Mean of Exponential samples: {np.mean(exp_samples):.4f} (Expected: {1/lambda_exp})")
```

**Central Limit Theorem**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrasi Central Limit Theorem
plt.figure(figsize=(15, 10))

# Berbagai distribusi populasi
distributions = [
    ('Uniform', stats.uniform(0, 1)),
    ('Exponential', stats.expon(scale=1)),
    ('Beta(2,5)', stats.beta(2, 5)),
    ('Gamma(2)', stats.gamma(2))
]

# Sample sizes
sample_sizes = [1, 2, 5, 30]

for i, (name, dist) in enumerate(distributions):
    for j, n in enumerate(sample_sizes):
        # Generate 10000 samples of size n
        samples = np.array([dist.rvs(size=n).mean() for _ in range(10000)])

        # Plot the histogram of sample means
        plt.subplot(4, 4, i*4 + j + 1)
        plt.hist(samples, bins=30, density=True, alpha=0.7)

        # Calculate and overlay the theoretical normal distribution
        if n > 1:  # Skip for n=1, since it's just the original distribution
            mean = dist.mean()
            std = dist.std() / np.sqrt(n)
            x = np.linspace(min(samples), max(samples), 100)
            plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)

        plt.title(f'{name}, n={n}')
        if i == 3:
            plt.xlabel('Sample Mean')
        if j == 0:
            plt.ylabel('Density')

plt.tight_layout()
plt.suptitle('Central Limit Theorem Demonstration', y=1.02, fontsize=16)
plt.show()

# Kesimpulan Central Limit Theorem:
# 1. Distribusi dari rata-rata sampel mendekati distribusi normal ketika ukuran sampel meningkat
# 2. Hal ini berlaku terlepas dari bentuk distribusi populasi aslinya
# 3. Standar deviasi distribusi sampling = standar deviasi populasi / sqrt(n)
```

#### Hypothesis_Testing_Basics

Pengujian hipotesis adalah metode untuk membuat keputusan berdasarkan data.

**Konsep Dasar**

Pengujian hipotesis melibatkan:

1. Hipotesis nol (H₀): Pernyataan "tidak ada efek" atau "tidak ada perbedaan"
2. Hipotesis alternatif (H₁ atau Hₐ): Pernyataan yang berlawanan dengan H₀
3. Statistik uji: Nilai yang dihitung dari data sampel
4. p-value: Probabilitas mendapatkan hasil yang sama atau lebih ekstrem dari yang diamati, jika H₀ benar
5. Tingkat signifikansi (α): Ambang batas untuk menolak H₀ (biasanya 0.05)

**One-Sample t-Test**

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Contoh: Apakah rata-rata tinggi mahasiswa berbeda dari 170 cm?
# H₀: μ = 170
# H₁: μ ≠ 170

# Data sampel
heights = np.array([173, 175, 180, 178, 177, 165, 167, 172, 168, 170])

# Menghitung statistik-t secara manual
mean = np.mean(heights)
std = np.std(heights, ddof=1)  # ddof=1 untuk sampel
n = len(heights)
se = std / np.sqrt(n)  # standard error
t_stat = (mean - 170) / se
df = n - 1  # degrees of freedom
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # two-tailed test

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
print(f"Standard Error: {se:.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Atau menggunakan fungsi bawaan
t_stat, p_value = stats.ttest_1samp(heights, 170)
print(f"t-statistic (scipy): {t_stat:.4f}")
print(f"p-value (scipy): {p_value:.4f}")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=6, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(170, color='red', linestyle='--', label='H₀: μ = 170')
plt.axvline(mean, color='green', linestyle='-', label=f'Sample mean: {mean:.2f}')
plt.title('Histogram of Heights with Null Hypothesis')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Kesimpulan
alpha = 0.05
if p_value < alpha:
    print(f"Karena p-value ({p_value:.4f}) < alpha ({alpha}), kita menolak H₀.")
    print("Ada cukup bukti untuk menyimpulkan bahwa rata-rata tinggi mahasiswa berbeda dari 170 cm.")
else:
    print(f"Karena p-value ({p_value:.4f}) ≥ alpha ({alpha}), kita tidak menolak H₀.")
    print("Tidak ada cukup bukti untuk menyimpulkan bahwa rata-rata tinggi mahasiswa berbeda dari 170 cm.")
```

**Two-Sample t-Test**

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Contoh: Apakah ada perbedaan rata-rata antara dua kelompok?
# H₀: μ₁ = μ₂
# H₁: μ₁ ≠ μ₂

# Data sampel
group1 = np.array([85, 90, 88, 92, 95, 87, 89, 91])
group2 = np.array([79, 78, 85, 80, 81, 86, 83, 84, 82, 81])

# Descriptive statistics
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)

print(f"Group 1: Mean = {mean1:.2f}, Std = {std1:.2f}, n = {n1}")
print(f"Group 2: Mean = {mean2:.2f}, Std = {std2:.2f}, n = {n2}")

# Menggunakan fungsi t-test dari scipy
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test (unequal variances)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.boxplot([group1, group2], labels=['Group 1', 'Group 2'])
plt.title('Box Plot Comparison')
plt.ylabel('Values')
plt.grid(alpha=0.3)
plt.show()

# Kesimpulan
alpha = 0.05
if p_value < alpha:
    print(f"Karena p-value ({p_value:.4f}) < alpha ({alpha}), kita menolak H₀.")
    print("Ada cukup bukti untuk menyimpulkan bahwa ada perbedaan rata-rata antara kedua kelompok.")
else:
    print(f"Karena p-value ({p_value:.4f}) ≥ alpha ({alpha}), kita tidak menolak H₀.")
    print("Tidak ada cukup bukti untuk menyimpulkan bahwa ada perbedaan rata-rata antara kedua kelompok.")
```

**Chi-Square Test for Independence**

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Contoh: Apakah ada hubungan antara jenis kelamin dan pemilihan program studi?
# H₀: Tidak ada hubungan (independen)
# H₁: Ada hubungan (dependen)

# Data: Tabel kontingensi
observed = np.array([
    [30, 15, 25],  # Laki-laki (Teknik, Bisnis, Sains)
    [20, 25, 15]   # Perempuan (Teknik, Bisnis, Sains)
])

# Chi-square test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print("Expected frequencies:")
print(expected)

# Visualisasi
plt.figure(figsize=(14, 6))

# Plot observed frequencies
plt.subplot(1, 2, 1)
df_observed = pd.DataFrame(observed,
                          index=['Laki-laki', 'Perempuan'],
                          columns=['Teknik', 'Bisnis', 'Sains'])
sns.heatmap(df_observed, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.title('Observed Frequencies')

# Plot expected frequencies
plt.subplot(1, 2, 2)
df_expected = pd.DataFrame(expected,
                          index=['Laki-laki', 'Perempuan'],
                          columns=['Teknik', 'Bisnis', 'Sains'])
sns.heatmap(df_expected, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False)
plt.title('Expected Frequencies (if independent)')

plt.tight_layout()
plt.show()

# Kesimpulan
alpha = 0.05
if p_value < alpha:
    print(f"Karena p-value ({p_value:.4f}) < alpha ({alpha}), kita menolak H₀.")
    print("Ada cukup bukti untuk menyimpulkan bahwa ada hubungan antara jenis kelamin dan pemilihan program studi.")
else:
    print(f"Karena p-value ({p_value:.4f}) ≥ alpha ({alpha}), kita tidak menolak H₀.")
    print("Tidak ada cukup bukti untuk menyimpulkan bahwa ada hubungan antara jenis kelamin dan pemilihan program studi.")
```

**ANOVA (Analysis of Variance)**

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Contoh: Apakah ada perbedaan rata-rata antara lebih dari dua kelompok?
# H₀: μ₁ = μ₂ = μ₃
# H₁: Setidaknya satu rata-rata kelompok berbeda

# Data sampel
group1 = np.array([85, 90, 88, 92, 95, 87, 89, 91])
group2 = np.array([79, 78, 85, 80, 81, 86, 83, 84, 82, 81])
group3 = np.array([75, 80, 82, 88, 84, 83, 80, 79, 81])

# Melakukan ANOVA dengan scipy
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualisasi
plt.figure(figsize=(10, 6))
data = [group1, group2, group3]
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Box Plot Comparison for ANOVA')
plt.ylabel('Values')
plt.grid(alpha=0.3)
plt.show()

# Jika p-value signifikan, lakukan post-hoc test (Tukey's HSD)
if p_value < 0.05:
    # Membuat dataframe untuk analisis post-hoc
    df = pd.DataFrame({
        'values': np.concatenate([group1, group2, group3]),
        'group': np.concatenate([
            np.repeat('Group 1', len(group1)),
            np.repeat('Group 2', len(group2)),
            np.repeat('Group 3', len(group3))
        ])
    })

    # Menggunakan statsmodels untuk post-hoc test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(endog=df['values'], groups=df['group'], alpha=0.05)
    print(tukey)

    # Visualisasi mean dan confidence intervals
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='group', y='values', data=df, join=False, ci=95)
    plt.title('Mean Values with 95% Confidence Intervals')
    plt.grid(alpha=0.3)
    plt.show()

# Kesimpulan
alpha = 0.05
if p_value < alpha:
    print(f"Karena p-value ({p_value:.4f}) < alpha ({alpha}), kita menolak H₀.")
    print("Ada cukup bukti untuk menyimpulkan bahwa setidaknya satu rata-rata kelompok berbeda.")
else:
    print(f"Karena p-value ({p_value:.4f}) ≥ alpha ({alpha}), kita tidak menolak H₀.")
    print("Tidak ada cukup bukti untuk menyimpulkan bahwa ada perbedaan rata-rata antar kelompok.")
```

#### Linear_Algebra_Basics

Linear algebra adalah fondasi matematika yang penting untuk machine learning dan data science.

**Vektor dan Matriks**

```python
import numpy as np

# Vektor
v = np.array([1, 2, 3])
print("Vektor v:")
print(v)
print(f"Dimensi: {v.shape}")

# Matriks
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("\nMatriks A:")
print(A)
print(f"Dimensi: {A.shape}")  # (rows, columns)

# Identitas
I = np.eye(3)  # 3x3 identity matrix
print("\nMatriks Identitas 3x3:")
print(I)

# Matriks Nol
zeros = np.zeros((2, 3))
print("\nMatriks Nol 2x3:")
print(zeros)

# Matriks Satu
ones = np.ones((2, 2))
print("\nMatriks Satu 2x2:")
print(ones)

# Transpose
A_T = A.T
print("\nTranspose dari A:")
print(A_T)
```

**Operasi Vektor dan Matriks**

```python
import numpy as np

# Vektor
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Penjumlahan dan pengurangan vektor
v_sum = v1 + v2
v_diff = v1 - v2
print("v1 + v2 =", v_sum)
print("v1 - v2 =", v_diff)

# Dot product (perkalian titik)
dot_product = np.dot(v1, v2)  # sama dengan v1 @ v2 atau v1.dot(v2)
print("v1 · v2 =", dot_product)

# Norm (panjang) vektor
v1_norm = np.linalg.norm(v1)
print("||v1|| =", v1_norm)

# Matriks
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Penjumlahan dan pengurangan matriks
C = A + B
D = A - B
print("\nA + B =")
print(C)
print("\nA - B =")
print(D)

# Perkalian matriks
E = A @ B  # sama dengan np.matmul(A, B) atau A.dot(B)
print("\nA × B =")
print(E)

# Perkalian elemen-wise
F = A * B
print("\nA * B (element-wise) =")
print(F)

# Determinan
det_A = np.linalg.det(A)
print("\ndet(A) =", det_A)

# Inverse
inv_A = np.linalg.inv(A)
print("\nA^(-1) =")
print(inv_A)

# Verifikasi inverse
print("\nA × A^(-1) =")
print(A @ inv_A)  # Harusnya mendekati matriks identitas
```

**Eigenvalues dan Eigenvectors**

```python
import numpy as np
import matplotlib.pyplot as plt

# Matriks untuk dihitung eigenvalues dan eigenvectors
A = np.array([[4, 2],
              [1, 3]])

# Menghitung eigenvalues dan eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matriks A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (kolom):")
print(eigenvectors)

# Verifikasi Ax = λx untuk setiap eigenvector
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    x_i = eigenvectors[:, i]
    Ax_i = A @ x_i
    lambda_x_i = lambda_i * x_i

    print(f"\nUntuk eigenvalue λ_{i+1} = {lambda_i}:")
    print(f"Eigenvector x_{i+1} = {x_i}")
    print(f"A·x_{i+1} = {Ax_i}")
    print(f"λ_{i+1}·x_{i+1} = {lambda_x_i}")
    print(f"Selisih: {np.linalg.norm(Ax_i - lambda_x_i)}")

# Visualisasi transformasi linear
# Membuat grid dari titik-titik
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.flatten(), Y.flatten()])

# Transformasi titik-titik dengan matriks A
transformed_points = A @ points

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(points[0], points[1], color='blue', alpha=0.5, label='Original')
plt.grid(True)
plt.title('Original Points')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(transformed_points[0], transformed_points[1], color='red', alpha=0.5, label='Transformed')
plt.grid(True)
plt.title('Transformed Points (Ax)')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Plot eigenvectors
for i in range(len(eigenvalues)):
    vec = eigenvectors[:, i]
    plt.subplot(1, 2, 1)
    plt.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1,
              fc='green', ec='green', label=f'Eigenvector {i+1}')

    plt.subplot(1, 2, 2)
    transformed_vec = A @ vec
    plt.arrow(0, 0, transformed_vec[0], transformed_vec[1], head_width=0.1, head_length=0.1,
              fc='green', ec='green')
    plt.arrow(0, 0, eigenvalues[i]*vec[0], eigenvalues[i]*vec[1], head_width=0.1, head_length=0.1,
              fc='purple', ec='purple', linestyle='--')

plt.tight_layout()
plt.show()
```

**Singular Value Decomposition (SVD)**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Singular Value Decomposition
# A = U · Σ · V^T
# di mana:
# - U adalah matriks ortogonal kiri
# - Σ adalah matriks diagonal dengan nilai singular
# - V^T adalah matriks ortogonal kanan yang ditranspose

# Contoh matriks
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Melakukan SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("Matriks A:")
print(A)
print("\nU (matriks orthogonal kiri):")
print(U)
print("\nΣ (nilai singular):")
print(np.diag(s))
print("\nV^T (matriks orthogonal kanan, ditranspose):")
print(Vt)

# Rekonstruksi
A_reconstructed = U @ np.diag(s) @ Vt
print("\nA rekonstruksi:")
print(A_reconstructed)
print("\nSelisih A dan rekonstruksi:")
print(np.linalg.norm(A - A_reconstructed))

# Aplikasi SVD: Kompresi gambar
# Load digit data
digits = load_digits()
X = digits.data.reshape(1797, 8, 8)

# Pilih satu gambar
img = X[0]

# SVD pada gambar
U, s, Vt = np.linalg.svd(img)

# Plot gambar asli
plt.figure(figsize=(15, 8))
plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Rekonstruksi dengan jumlah nilai singular yang berbeda
for i, k in enumerate([1, 2, 3, 5, 7]):
    # Rekonstruksi dengan k komponen
    img_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    plt.subplot(2, 4, i + 2)
    plt.imshow(img_approx, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Plot nilai singular
plt.figure(figsize=(10, 6))
plt.plot(s, 'o-')
plt.title('Singular Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

# Fundamental Data Science

## Calculus Basics

Kalkulus adalah cabang matematika yang mempelajari perubahan, seperti kemiringan kurva atau laju perubahan suatu kuantitas. Dalam data science, kalkulus sangat penting untuk memahami dan mengoptimalkan algoritma machine learning.

### 1. Turunan (Derivatives)

Turunan mengukur laju perubahan suatu fungsi terhadap variabelnya. Ini adalah konsep dasar dalam gradient descent, algoritma optimasi yang digunakan di hampir semua model machine learning.

**Rumus Dasar:**

-   Turunan dari f(x) ditulis sebagai f'(x) atau df/dx
-   Untuk fungsi f(x) = x², turunannya adalah f'(x) = 2x

**Contoh Kode Python:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Mendefinisikan fungsi
def f(x):
    return x**2

# Mendefinisikan turunan secara manual
def df(x):
    return 2*x

# Membuat data untuk plot
x = np.linspace(-5, 5, 100)
y = f(x)
dy = df(x)

# Visualisasi
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.plot(x, dy, 'r-', label='f\'(x) = 2x')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Fungsi dan Turunannya')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 2. Integral (Integrals)

Integral adalah kebalikan dari turunan, yang mengukur area di bawah kurva. Dalam data science, integral digunakan dalam probabilitas dan statistik.

**Rumus Dasar:**

-   Integral dari f(x) ditulis sebagai ∫f(x)dx
-   Untuk f(x) = x², integral tak tentu adalah ∫x²dx = (1/3)x³ + C

**Contoh Kode Python:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Mendefinisikan fungsi
def f(x):
    return x**2

# Area di bawah kurva dari 0 sampai 2
area, error = integrate.quad(f, 0, 2)
print(f"Area di bawah kurva x² dari 0 sampai 2: {area}")  # Seharusnya 8/3

# Visualisasi
x = np.linspace(0, 2, 100)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.fill_between(x, y, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f'Integral dari x² (0 sampai 2) = {area:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 3. Partial Derivatives

Dalam machine learning, kebanyakan model melibatkan banyak variabel, sehingga turunan parsial sangat penting untuk algoritma optimasi.

**Contoh Fungsi Biaya (Cost Function) Linear Regression:**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Membuat data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fungsi biaya untuk linear regression
def cost_function(theta0, theta1, X, y):
    m = len(y)
    predictions = theta0 + theta1 * X
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

# Membuat grid untuk visualisasi
theta0_vals = np.linspace(0, 10, 100)
theta1_vals = np.linspace(0, 5, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
cost_grid = np.zeros(theta0_grid.shape)

# Menghitung biaya untuk setiap kombinasi parameter
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        cost_grid[j, i] = cost_function(theta0_vals[i], theta1_vals[j], X, y)

# Visualisasi
fig = plt.figure(figsize=(12, 6))

# Plot permukaan biaya
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(theta0_grid, theta1_grid, cost_grid, cmap='viridis', alpha=0.8)
ax1.set_xlabel('theta0')
ax1.set_ylabel('theta1')
ax1.set_zlabel('Cost')
ax1.set_title('Permukaan Fungsi Biaya')

# Plot kontur biaya
ax2 = fig.add_subplot(122)
contour = ax2.contour(theta0_grid, theta1_grid, cost_grid, 20, cmap='viridis')
ax2.set_xlabel('theta0')
ax2.set_ylabel('theta1')
ax2.set_title('Kontur Fungsi Biaya')
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()
```

### 4. Gradient Descent

Gradient descent adalah algoritma optimasi yang menggunakan turunan untuk menemukan nilai minimum dari fungsi. Ini sangat penting dalam machine learning untuk menemukan parameter optimal.

**Contoh Kode untuk Linear Regression:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Data sederhana
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Menambahkan kolom untuk bias (intercept)
X_b = np.c_[np.ones((100, 1)), X]

# Gradient Descent untuk Linear Regression
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = np.zeros(n_iterations)
    theta_history = np.zeros((n_iterations, 2))

    for it in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
        theta_history[it,:] = theta.T
        predictions = X_b.dot(theta)
        cost_history[it] = (1/m) * np.sum(np.square(predictions - y))

    return theta, cost_history, theta_history

# Parameter awal
theta = np.random.randn(2,1)

# Jalankan gradient descent
learning_rate = 0.1
n_iterations = 100
theta_final, cost_history, theta_history = gradient_descent(X_b, y, theta, learning_rate, n_iterations)

print(f"Parameter akhir: theta0 = {theta_final[0][0]:.4f}, theta1 = {theta_final[1][0]:.4f}")

# Visualisasi hasil
plt.figure(figsize=(16, 6))

# Plot data dan garis hasil regresi
plt.subplot(1, 2, 1)
plt.scatter(X, y)
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_final)
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Prediksi")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression dengan Gradient Descent")
plt.legend()

# Plot cost history
plt.subplot(1, 2, 2)
plt.plot(range(n_iterations), cost_history)
plt.xlabel("Iterasi")
plt.ylabel("Cost")
plt.title("Evolusi Fungsi Biaya")

plt.tight_layout()
plt.show()
```

## Mathematics and Statistics

### 1. Descriptive Statistics

Statistik deskriptif adalah metode untuk mengorganisir, merangkum, dan menyajikan data dengan cara yang informatif.

**Ukuran Tendensi Sentral:**

-   Mean (rata-rata): ukuran pusat data
-   Median: nilai tengah dalam dataset terurut
-   Mode: nilai yang paling sering muncul

**Ukuran Dispersi:**

-   Range: selisih nilai maksimum dan minimum
-   Varians: rata-rata deviasi kuadrat dari mean
-   Standar deviasi: akar kuadrat dari varians

**Contoh Kode:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membuat dataset contoh
np.random.seed(42)
data = np.random.normal(loc=70, scale=10, size=1000)

# Menghitung statistik deskriptif
mean_val = np.mean(data)
median_val = np.median(data)
std_val = np.std(data)
min_val = np.min(data)
max_val = np.max(data)
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1

# Membuat DataFrame untuk statistik deskriptif
stats_df = pd.DataFrame({
    'Statistik': ['Mean', 'Median', 'Standar Deviasi', 'Minimum', 'Maximum', 'Q1', 'Q3', 'IQR'],
    'Nilai': [mean_val, median_val, std_val, min_val, max_val, q1, q3, iqr]
})

print(stats_df)

# Visualisasi
plt.figure(figsize=(12, 6))

# Histogram dengan density plot
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True)
plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
plt.title('Distribusi Data')
plt.legend()

# Box plot
plt.subplot(1, 2, 2)
sns.boxplot(x=data)
plt.title('Box Plot')
plt.xlabel('Nilai')

plt.tight_layout()
plt.show()
```

### 2. Probability Basics

Probabilitas adalah ukuran kemungkinan suatu peristiwa terjadi. Ini adalah dasar untuk memahami statistik inferensial dan banyak algoritma machine learning.

**Konsep Dasar:**

-   Probabilitas berkisar antara 0 (tidak mungkin) hingga 1 (pasti)
-   P(A ∪ B) = P(A) + P(B) - P(A ∩ B) (Hukum Penjumlahan)
-   P(A ∩ B) = P(A) × P(B|A) (Hukum Perkalian)
-   P(A|B) = P(A ∩ B) / P(B) (Probabilitas Bersyarat)

**Contoh Kode untuk Simulasi Probabilitas:**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulasi pelemparan dadu
np.random.seed(42)
n_trials = 10000
dice_results = np.random.randint(1, 7, n_trials)

# Menghitung probabilitas empiris
probabilities = np.bincount(dice_results)[1:] / n_trials

# Visualisasi
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(1, 7), y=probabilities)
plt.axhline(y=1/6, color='r', linestyle='--', label='Probabilitas Teoretis (1/6)')
plt.xlabel('Hasil Dadu')
plt.ylabel('Probabilitas')
plt.title(f'Probabilitas Empiris dari {n_trials} Pelemparan Dadu')
plt.legend()
plt.ylim(0, 0.25)
plt.show()

# Demonstrasi hukum probabilitas bersyarat dengan kartu
suits = ['Hati', 'Wajik', 'Sekop', 'Keriting']
ranks = list(range(2, 11)) + ['J', 'Q', 'K', 'A']
cards = [(s, r) for s in suits for r in ranks]

# Fungsi untuk menghitung probabilitas
def probability_experiment():
    # Simulasi pengambilan kartu
    n_simulations = 100000

    # Menghitung P(A): Probabilitas mendapatkan kartu hati
    count_hearts = 0

    # Menghitung P(B): Probabilitas mendapatkan kartu As
    count_aces = 0

    # Menghitung P(A ∩ B): Probabilitas mendapatkan As hati
    count_heart_ace = 0

    for _ in range(n_simulations):
        card = cards[np.random.randint(0, len(cards))]
        if card[0] == 'Hati':
            count_hearts += 1
        if card[1] == 'A':
            count_aces += 1
        if card[0] == 'Hati' and card[1] == 'A':
            count_heart_ace += 1

    p_hearts = count_hearts / n_simulations
    p_aces = count_aces / n_simulations
    p_heart_ace = count_heart_ace / n_simulations

    # Menghitung P(A|B): Probabilitas kartu hati diberikan bahwa kartu adalah As
    p_heart_given_ace = p_heart_ace / p_aces if p_aces > 0 else 0

    # Hasil teoretis
    p_hearts_theory = 13/52  # 13 kartu hati dari 52 kartu
    p_aces_theory = 4/52     # 4 As dari 52 kartu
    p_heart_ace_theory = 1/52  # 1 As hati dari 52 kartu
    p_heart_given_ace_theory = 1/4  # 1 hati dari 4 As

    results = pd.DataFrame({
        'Probabilitas': ['P(Hati)', 'P(As)', 'P(Hati ∩ As)', 'P(Hati | As)'],
        'Empiris': [p_hearts, p_aces, p_heart_ace, p_heart_given_ace],
        'Teoretis': [p_hearts_theory, p_aces_theory, p_heart_ace_theory, p_heart_given_ace_theory]
    })

    return results

prob_results = probability_experiment()
print("\nHasil Eksperimen Probabilitas Kartu:")
print(prob_results)
```

### 3. Probability Distributions

Distribusi probabilitas adalah model yang menghubungkan nilai variabel acak dengan probabilitasnya. Beberapa distribusi penting dalam data science termasuk:

**Distribusi Diskrit:**

-   Distribusi Binomial: menggambarkan jumlah sukses dalam n percobaan
-   Distribusi Poisson: menggambarkan jumlah kejadian dalam interval tetap

**Distribusi Kontinu:**

-   Distribusi Normal: pola berbentuk lonceng yang banyak ditemui di alam
-   Distribusi Eksponensial: menggambarkan waktu antar kejadian

**Contoh Kode:**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.figure(figsize=(18, 12))

# 1. Distribusi Binomial
n_trials = 20
p_success = 0.3
x = np.arange(0, n_trials + 1)
binomial = stats.binom.pmf(x, n_trials, p_success)

plt.subplot(2, 2, 1)
plt.bar(x, binomial)
plt.title(f'Distribusi Binomial (n={n_trials}, p={p_success})')
plt.xlabel('Jumlah Sukses')
plt.ylabel('Probabilitas')

# 2. Distribusi Poisson
lambda_val = 3
x = np.arange(0, 15)
poisson = stats.poisson.pmf(x, lambda_val)

plt.subplot(2, 2, 2)
plt.bar(x, poisson)
plt.title(f'Distribusi Poisson (λ={lambda_val})')
plt.xlabel('Jumlah Kejadian')
plt.ylabel('Probabilitas')

# 3. Distribusi Normal
x = np.linspace(-5, 5, 1000)
norm_pdf = stats.norm.pdf(x, loc=0, scale=1)

plt.subplot(2, 2, 3)
plt.plot(x, norm_pdf)
plt.fill_between(x, norm_pdf, alpha=0.3)
plt.title('Distribusi Normal Standar (μ=0, σ=1)')
plt.xlabel('Nilai')
plt.ylabel('Densitas Probabilitas')

# 4. Distribusi Eksponensial
lambda_val = 0.5
x = np.linspace(0, 10, 1000)
exp_pdf = stats.expon.pdf(x, scale=1/lambda_val)

plt.subplot(2, 2, 4)
plt.plot(x, exp_pdf)
plt.fill_between(x, exp_pdf, alpha=0.3)
plt.title(f'Distribusi Eksponensial (λ={lambda_val})')
plt.xlabel('Nilai')
plt.ylabel('Densitas Probabilitas')

plt.tight_layout()
plt.show()

# Demonstrasi Teorema Limit Pusat
plt.figure(figsize=(15, 10))

for i, sample_size in enumerate([1, 2, 5, 30]):
    # Mengambil sampel dari distribusi seragam dan menghitung rata-ratanya
    np.random.seed(42)
    samples = np.random.uniform(0, 1, size=(10000, sample_size))
    sample_means = samples.mean(axis=1)

    plt.subplot(2, 2, i+1)
    sns.histplot(sample_means, kde=True, stat="density")
    plt.title(f'Rata-rata {sample_size} sampel dari Distribusi Seragam')
    plt.xlabel('Rata-rata Sampel')
    plt.ylabel('Densitas')

    # Overlay distribusi normal dengan parameter yang sama
    x = np.linspace(min(sample_means), max(sample_means), 1000)
    plt.plot(x, stats.norm.pdf(x, np.mean(sample_means), np.std(sample_means)),
             'r-', linewidth=2, label='Normal PDF')
    if i == 3:
        plt.legend()

plt.suptitle('Demonstrasi Teorema Limit Pusat', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
```

### 4. Hypothesis Testing Basics

Pengujian hipotesis adalah metode statistik untuk membuat keputusan berdasarkan data. Ini melibatkan perbandingan hipotesis nol (biasanya tidak ada efek) dengan hipotesis alternatif.

**Langkah-langkah:**

1. Tentukan hipotesis nol (H₀) dan alternatif (H₁)
2. Pilih tingkat signifikansi (α)
3. Hitung statistik uji
4. Tentukan p-value
5. Bandingkan p-value dengan α dan buat keputusan

**Contoh Kode untuk Uji t:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Membuat dua sampel data
np.random.seed(42)
sample1 = np.random.normal(loc=50, scale=5, size=100)  # Kelompok kontrol
sample2 = np.random.normal(loc=52, scale=5, size=100)  # Kelompok treatment

# Melakukan uji t independen
t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=True)

print(f"Statistik t: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretasi
alpha = 0.05
if p_value < alpha:
    print(f"Menolak hipotesis nol (p={p_value:.4f} < {alpha})")
    print("Ada perbedaan signifikan antara kedua kelompok")
else:
    print(f"Gagal menolak hipotesis nol (p={p_value:.4f} >= {alpha})")
    print("Tidak ada cukup bukti untuk menyatakan perbedaan signifikan")

# Visualisasi
plt.figure(figsize=(12, 6))

# Histograms
plt.subplot(1, 2, 1)
sns.histplot(sample1, kde=True, color='blue', alpha=0.5, label='Kelompok 1')
sns.histplot(sample2, kde=True, color='red', alpha=0.5, label='Kelompok 2')
plt.axvline(np.mean(sample1), color='blue', linestyle='--', label=f'Mean 1: {np.mean(sample1):.2f}')
plt.axvline(np.mean(sample2), color='red', linestyle='--', label=f'Mean 2: {np.mean(sample2):.2f}')
plt.title('Distribusi Data Kedua Kelompok')
plt.legend()

# Box plot
plt.subplot(1, 2, 2)
data = pd.DataFrame({
    'Kelompok': ['Kelompok 1']*len(sample1) + ['Kelompok 2']*len(sample2),
    'Nilai': np.concatenate([sample1, sample2])
})
sns.boxplot(x='Kelompok', y='Nilai', data=data)
plt.title(f'Box Plot Perbandingan (p-value: {p_value:.4f})')

plt.tight_layout()
plt.show()

# Mendemonstrasikan interval kepercayaan
from scipy import stats

# Menghitung interval kepercayaan 95% untuk perbedaan rata-rata
conf_interval = stats.t.interval(
    alpha=0.95,  # 95% confidence interval
    df=len(sample1) + len(sample2) - 2,  # degrees of freedom
    loc=np.mean(sample2) - np.mean(sample1),  # point estimate (difference in means)
    scale=np.sqrt(((len(sample1)-1) * np.var(sample1, ddof=1) +
                  (len(sample2)-1) * np.var(sample2, ddof=1)) /
                  (len(sample1) + len(sample2) - 2) *
                  (1/len(sample1) + 1/len(sample2)))  # standard error
)

print(f"\nInterval Kepercayaan 95% untuk perbedaan rata-rata: {conf_interval}")

# Visualisasi interval kepercayaan
plt.figure(figsize=(10, 6))
diff_mean = np.mean(sample2) - np.mean(sample1)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=diff_mean, color='r', linestyle='-', label=f'Perbedaan Mean: {diff_mean:.2f}')
plt.axvline(x=conf_interval[0], color='g', linestyle='--', label=f'Batas Bawah: {conf_interval[0]:.2f}')
plt.axvline(x=conf_interval[1], color='g', linestyle='--', label=f'Batas Atas: {conf_interval[1]:.2f}')
plt.fill_between([conf_interval[0], conf_interval[1]], -0.1, 0.1, color='g', alpha=0.2)
plt.title('Interval Kepercayaan 95% untuk Perbedaan Rata-rata')
plt.xlabel('Perbedaan (Kelompok 2 - Kelompok 1)')
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()
```

### 5. Linear Algebra Basics

Aljabar linear adalah cabang matematika yang mempelajari vektor, matriks, dan transformasi linear. Ini sangat penting dalam data science untuk pengolahan data multidimensi dan machine learning.

**Konsep Dasar:**

-   Vektor: array satu dimensi dari angka
-   Matriks: array dua dimensi dari angka
-   Operasi matriks: penjumlahan, perkalian, transpose, invers

**Contoh Kode:**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Operasi Vektor
v1 = np.array([2, 3, 4])
v2 = np.array([1, 0, 2])

# Penjumlahan vektor
v_sum = v1 + v2

# Perkalian skalar
v_scaled = 2 * v1

# Dot product
dot_product = np.dot(v1, v2)

print("Vektor 1:", v1)
print("Vektor 2:", v2)
print("Penjumlahan:", v_sum)
print("Skalar × Vektor 1:", v_scaled)
print("Dot Product:", dot_product)

# Visualisasi vektor 2D
plt.figure(figsize=(10, 8))
plt.quiver(0, 0, 3, 4, angles='xy', scale_units='xy', scale=1, color='r', label='Vektor [3,4]')
plt.quiver(0, 0, 1, 2, angles='xy', scale_units='xy', scale=1, color='b', label='Vektor [1,2]')
plt.quiver(0, 0, 4, 6, angles='xy', scale_units='xy', scale=1, color='g', label='Penjumlahan [4,6]')
plt.xlim(-1, 6)
plt.ylim(-1, 8)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.legend()
plt.title('Visualisasi Vektor 2D')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

# 2. Operasi Matriks
A = np.array([[1, 2, 3],
             [4, 5, 6]])

B = np.array([[7, 8],
             [9, 10],
             [11, 12]])

# Perkalian matriks
C = np.dot(A, B)

print("\nMatriks A:")
print(A)
print("\nMatriks B:")
print(B)
print("\nPerkalian matriks A·B:")
print(C)

# 3. Eigenvalues & Eigenvectors
square_matrix = np.array([[4, 2],
                          [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(square_matrix)

print("\nMatriks:")
print(square_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Visualisasi transformasi linear
def plot_transformation(matrix):
    # Titik-titik asli
    x = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])

    # Transformasi
    y = np.dot(matrix, x)

    plt.figure(figsize=(10, 10))

    # Plot titik asli
    plt.plot(x[0], x[1], 'b-', label='Objek Asli')
    plt.fill(x[0], x[1], 'b', alpha=0.2)

    # Plot titik hasil transformasi
    plt.plot(y[0], y[1], 'r-', label='Hasil Transformasi')
    plt.fill(y[0], y[1], 'r', alpha=0.2)

    # Visualisasi eigenvektor
    for i in range(len(eigenvalues)):
        eigv = eigenvectors[:, i]
        plt.quiver(0, 0, eigv[0], eigv[1], angles='xy', scale_units='xy',
                  scale=1, color='g', width=0.01, label=f'Eigenvektor {i+1}')

    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlim(-2, 5)
    plt.ylim(-2, 5)
    plt.legend()
    plt.title('Transformasi Linear dan Eigenvektor')
    plt.axis('equal')
    plt.show()

plot_
```

## 1. Feature Engineering yang Lebih Mendalam

Feature Engineering adalah salah satu tahap paling krusial dalam pipeline data science yang dapat sangat meningkatkan performa model.

### Teknik-teknik Feature Engineering

#### a. Transformasi Variabel

-   **Log Transform**: Mengatasi skewness pada distribusi data
    ```python
    df['log_income'] = np.log1p(df['income'])  # log(1+x) untuk menghindari log(0)
    ```
-   **Box-Cox Transform**: Transformasi yang lebih umum untuk normalisasi
    ```python
    from scipy import stats
    df['boxcox_income'], lambda_param = stats.boxcox(df['income'])
    ```
-   **Yeo-Johnson Transform**: Alternatif Box-Cox yang bekerja untuk nilai negatif
    ```python
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    df['transformed_income'] = pt.fit_transform(df[['income']])
    ```

#### b. Categorical Encoding yang Lebih Kompleks

-   **One-Hot Encoding**: Untuk variabel nominal

    ```python
    df_encoded = pd.get_dummies(df, columns=['city', 'color'])
    ```

-   **Label Encoding**: Untuk variabel ordinal

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['education_encoded'] = le.fit_transform(df['education_level'])
    ```

-   **Target Encoding**: Mengganti kategori dengan rata-rata target untuk kategori tersebut

    ```python
    means = df.groupby('city')['target'].mean()
    df['city_encoded'] = df['city'].map(means)
    ```

-   **Weight of Evidence (WoE)**: Berguna untuk model regresi logistik

    ```python
    def calculate_woe(df, feature, target):
        groups = df.groupby(feature)[target].agg(['sum', 'count'])
        groups['non_target'] = groups['count'] - groups['sum']
        groups['woe'] = np.log((groups['sum'] / groups['sum'].sum()) /
                             (groups['non_target'] / groups['non_target'].sum()))
        return groups['woe'].to_dict()

    woe_dict = calculate_woe(df, 'city', 'is_customer')
    df['city_woe'] = df['city'].map(woe_dict)
    ```

#### c. Feature Extraction

-   **Polynomial Features**: Menangkap interaksi antar fitur
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[['income', 'age']])
    ```
-   **Principal Component Analysis (PCA)**
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    ```
-   **t-SNE**: Untuk visualisasi data berdimensi tinggi
    ```python
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled_features)
    ```

#### d. Time-based Features

-   **Ekstraksi fitur dari timestamp**
    ```python
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    ```
-   **Lag Features**: Fitur dari periode sebelumnya
    ```python
    df['sales_lag1'] = df.groupby('store_id')['sales'].shift(1)
    df['sales_lag7'] = df.groupby('store_id')['sales'].shift(7)  # lag 7 hari
    ```
-   **Rolling Window Features**
    ```python
    df['sales_rolling_mean_7d'] = df.groupby('store_id')['sales'].rolling(7).mean().reset_index(0, drop=True)
    df['sales_rolling_std_7d'] = df.groupby('store_id')['sales'].rolling(7).std().reset_index(0, drop=True)
    ```

#### e. Text Features

-   **Bag of Words**
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    ```
-   **TF-IDF**
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    ```
-   **Word Embeddings**: Word2Vec, GloVe, BERT
    ```python
    # Menggunakan pre-trained embeddings dengan gensim
    import gensim.downloader as api
    word_vectors = api.load("glove-wiki-gigaword-100")
    ```

### Feature Selection

-   **Filter Methods**

    -   Variance Threshold

        ```python
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.1)
        X_selected = selector.fit_transform(X)
        ```

    -   Correlation-based

        ```python
        # Menghapus fitur dengan korelasi tinggi
        def correlation_filter(X, threshold=0.8):
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            return X.drop(to_drop, axis=1)

        X_selected = correlation_filter(X, threshold=0.8)
        ```

    -   Statistical Tests

        ```python
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=10)
        X_selected = selector.fit_transform(X, y)
        ```

-   **Wrapper Methods**

    -   Recursive Feature Elimination
        ```python
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier()
        selector = RFE(estimator, n_features_to_select=10)
        X_selected = selector.fit_transform(X, y)
        ```

-   **Embedded Methods**

    -   Lasso Regression (L1 regularization)
        ```python
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
        model.fit(X, y)
        # Fitur dengan koefisien non-zero
        important_features = X.columns[model.coef_ != 0]
        ```
    -   Feature Importance dari Tree-based Models
        ```python
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X, y)
        importances = model.feature_importances_
        # Sorting feature berdasarkan importance
        indices = np.argsort(importances)[::-1]
        ```

## 2. Advanced Machine Learning Techniques

### Ensemble Methods

Ensemble methods menggabungkan beberapa model untuk menghasilkan prediksi yang lebih akurat.

#### a. Bagging (Bootstrap Aggregating)

-   **Random Forest**: Ensemble dari decision trees
    ```python
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    ```

#### b. Boosting

-   **AdaBoost**
    ```python
    from sklearn.ensemble import AdaBoostClassifier
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
    adaboost.fit(X_train, y_train)
    ```
-   **Gradient Boosting**
    ```python
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    ```
-   **XGBoost**: Implementasi gradient boosting yang efisien
    ```python
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    ```
-   **LightGBM**: Implementasi gradient boosting yang lebih cepat
    ```python
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)
    ```
-   **CatBoost**: Menangani categorical features secara otomatis
    ```python
    import catboost as cb
    cat_model = cb.CatBoostClassifier(iterations=100, random_seed=42)
    cat_model.fit(X_train, y_train)
    ```

#### c. Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train, y_train)
```

### Hyperparameter Tuning

Teknik untuk menemukan parameter optimal untuk model machine learning.

#### a. Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
```

#### b. Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

#### c. Bayesian Optimization

```python
# Menggunakan library hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
    'max_depth': hp.quniform('max_depth', 3, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1)
}

def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']) if params['max_depth'] > 0 else None,
        'min_samples_split': int(params['min_samples_split'])
    }

    clf = RandomForestClassifier(**params, random_state=42)
    return {'loss': -np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')), 'status': STATUS_OK}

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
```

## 3. Time Series Analysis

Time series adalah data yang diurutkan berdasarkan waktu. Time series analysis sangat penting untuk forecasting dan pemahaman tren.

### Components of Time Series

-   **Trend**: Pola jangka panjang
-   **Seasonality**: Pola siklik yang terjadi pada interval tertentu
-   **Cyclical**: Fluktuasi tidak teratur dalam data
-   **Irregularity**: Fluktuasi acak yang tidak terprediksi

### Teknik-teknik Time Series

#### a. Moving Average

```python
# Simple Moving Average
df['SMA_7'] = df['sales'].rolling(window=7).mean()

# Exponential Moving Average
df['EMA_7'] = df['sales'].ewm(span=7, adjust=False).mean()
```

#### b. Time Series Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['sales'], model='multiplicative', period=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
df['sales'].plot(ax=ax1)
ax1.set_title('Original')
trend.plot(ax=ax2)
ax2.set_title('Trend')
seasonal.plot(ax=ax3)
ax3.set_title('Seasonality')
residual.plot(ax=ax4)
ax4.set_title('Residuals')
plt.tight_layout()
```

#### c. ARIMA (AutoRegressive Integrated Moving Average)

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model: ARIMA(p,d,q)
model = ARIMA(df['sales'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=30)
```

#### d. SARIMA (Seasonal ARIMA)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA model: SARIMA(p,d,q)(P,D,Q,s)
model = SARIMAX(df['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=30)
```

#### e. Prophet

```python
from fbprophet import Prophet

# Prepare dataframe for Prophet (needs 'ds' and 'y' columns)
prophet_df = df.rename(columns={'date': 'ds', 'sales': 'y'})

# Fit model
model = Prophet()
model.fit(prophet_df)

# Create future dataframe for predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
fig_components = model.plot_components(forecast)
```

## 4. Deep Learning

Deep learning adalah subset dari machine learning yang menggunakan neural networks dengan banyak layers untuk menganalisis berbagai jenis data.

### Artificial Neural Networks (ANN)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # untuk klasifikasi biner
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
```

### Convolutional Neural Networks (CNN)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # untuk 10 kelas (MNIST)
])

# Compile model
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)
```

### Recurrent Neural Networks (RNN)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Untuk sequence data (misalnya NLP)
model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=100),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # untuk sentimen analisis biner
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## 5. Explainable AI (XAI)

XAI adalah teknik untuk memahami dan menjelaskan keputusan model machine learning.

### SHAP (SHapley Additive exPlanations)

```python
import shap

# Jelaskan model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot summary
shap.summary_plot(shap_values, X_test)

# Plot untuk observasi pertama
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime.lime_tabular import LimeTabularExplainer

# Inisialisasi explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Negative', 'Positive'],
    discretize_continuous=True
)

# Jelaskan prediksi
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# Visualisasi
exp.show_in_notebook()
```

### Partial Dependence Plots (PDP)

```python
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Plot PDP
features = [0, 1, (0, 1)]  # Indeks fitur yang ingin dilihat
plot_partial_dependence(
    model, X_train, features,
    feature_names=X_train.columns,
    grid_resolution=20
)
plt.tight_layout()
plt.show()
```

## 6. Model Deployment

Deployment memungkinkan model machine learning digunakan di aplikasi produksi.

### Flask API

```python
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

### Streamlit Web App

```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Judul
st.title('Model Prediksi Diabetes')

# Inputs
st.header('Masukkan Data Pasien')
glucose = st.slider('Glukosa', 0, 200, 100)
bp = st.slider('Tekanan Darah', 0, 150, 70)
insulin = st.slider('Insulin', 0, 200, 80)
bmi = st.slider('BMI', 10.0, 50.0, 25.0)

# Prediction button
if st.button('Prediksi'):
    # Membuat dataframe dari input
    data = pd.DataFrame({
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'Insulin': [insulin],
        'BMI': [bmi]
    })

    # Prediksi
    prediction = model.predict(data)

    # Output
    if prediction[0] == 1:
        st.error('Pasien kemungkinan menderita diabetes')
    else:
        st.success('Pasien kemungkinan tidak menderita diabetes')
```

### Docker Containerization

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY model.pkl .
COPY app.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
```

## 7. MLOps (Machine Learning Operations)

MLOps adalah praktik DevOps yang diterapkan untuk workload machine learning.

### Automated Testing

```python
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_model_performance():
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Assert minimum performance
    assert accuracy > 0.8, f"Model accuracy too low: {accuracy}"
```

### Model Versioning

```python
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("diabetes-prediction")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "random-forest-model")
```

### Continuous Training

```python
# Buat script untuk retraining otomatis
# retrain.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(data_path, model_path):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrain model with new data')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--model', type=str, default='model.pkl', help='Path to save model')
    args = parser.parse_args()

    train_model(args.data, args.model)
```

```bash
# Untuk CI/CD pipeline (misalnya dengan GitHub Actions):
name: Model Retraining

on:
  schedule:
    - cron: '0 0 * * 0'  # Run weekly
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Download latest data
      run: |
        python download_data.py --output data.csv

    - name: Retrain model
      run: |
        python retrain.py --data data.csv --model model.pkl

    - name: Upload model artifact
      uses: actions/upload-artifact@v2
      with:
        name: model
        path: model.pkl
```

## 8. Data Engineering untuk Data Science

### Data Pipelines

```python
# Menggunakan Apache Airflow untuk data pipelines
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_scientist',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='An ML training pipeline DAG',
    schedule_interval=timedelta(days=1),
)

def extract_data():
    # Extract data from database or API
    pass

def transform_data():
    # Clean and transform data
    pass

def train_model():
    # Train and save model
    pass

def evaluate_model():
    # Evaluate model performance
    pass

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

extract_task >> transform_task >> train_task >> evaluate_task
```

### Data Versioning

```python
# Menggunakan DVC (Data Version Control)
# Inisialisasi DVC
# $ dvc init

# Track data files
# $ dvc add data.csv

# Commit ke Git
# $ git add data.csv.dvc .gitignore
# $ git commit -m "Add data tracking"

# Push ke remote storage
# $ dvc remote add -d myremote s3://my-bucket/dvc-storage
# $ dvc push
```

### Big Data Processing

```python
# Menggunakan PySpark untuk data berskala besar
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Inisialisasi Spark Session
spark = SparkSession.builder \
    .appName("BigDataMLPipeline") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Baca data
df = spark.read.csv("hdfs://path/to/bigdata.csv", header=True, inferSchema=True)

# Preprocessing
feature_cols = [col for col in df.columns if col != "target"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# Split data
train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Train model
rf = RandomForestClassifier(labelCol="target", featuresCol="features", numTrees=100)
model = rf.fit(train_df)

# Evaluasi
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="target")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# Simpan model
model.write().overwrite().save("hdfs://path/to/model")
```

### Distributed Data Processing dengan Dask

Dask adalah library Python untuk komputasi paralel yang bekerja dengan ekosistem PyData (NumPy, Pandas, Scikit-learn).

```python
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

# Mulai Dask client
client = Client()  # untuk komputasi lokal
# atau
# client = Client('scheduler-address:8786')  # untuk cluster

# Baca data besar dengan Dask DataFrame
df = dd.read_csv('s3://bucket/big-data-*.csv')

# Operasi seperti Pandas
result = df.groupby('category').agg({'value': ['mean', 'std']})

# Hitung hasil (lazy evaluation)
result = result.compute()
```

## 8. Advanced Data Visualization

Visualisasi data yang efektif sangat penting untuk exploratory data analysis dan komunikasi hasil.

### Interactive Visualizations dengan Plotly

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Data
df = pd.read_csv('sales_data.csv')

# Visualisasi interaktif sederhana
fig = px.scatter(df, x="price", y="sales", color="category",
                 size="market_size", hover_name="product",
                 log_x=True, size_max=60)
fig.show()

# Custom layout
fig.update_layout(
    title="Sales Analysis by Price and Category",
    xaxis_title="Price (log scale)",
    yaxis_title="Sales Volume",
    legend_title="Product Category"
)

# Visualisasi lebih kompleks
fig = go.Figure()

for category in df['category'].unique():
    subset = df[df['category'] == category]
    fig.add_trace(go.Scatter(
        x=subset["date"],
        y=subset["sales"],
        mode='lines+markers',
        name=category
    ))

fig.update_layout(
    title="Sales Trends by Category",
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified"
)

fig.show()
```

### Dashboard dengan Dash

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('sales_data.csv')

# Initialize app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Sales Dashboard"),

    html.Div([
        html.Div([
            html.Label("Select Category:"),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': i, 'value': i} for i in df['category'].unique()],
                value=df['category'].unique()[0]
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['date'].min(),
                end_date=df['date'].max(),
                max_date_allowed=df['date'].max()
            ),
        ], style={'width': '70%', 'display': 'inline-block'})
    ]),

    html.Div([
        dcc.Graph(id='sales-time-series')
    ]),

    html.Div([
        dcc.Graph(id='products-bar-chart')
    ])
])

# Define callbacks
@app.callback(
    Output('sales-time-series', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_time_series(selected_category, start_date, end_date):
    filtered_df = df[(df['category'] == selected_category) &
                     (df['date'] >= start_date) &
                     (df['date'] <= end_date)]

    fig = px.line(filtered_df, x='date', y='sales',
                  title=f'Sales Trend: {selected_category}')
    return fig

@app.callback(
    Output('products-bar-chart', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_bar_chart(selected_category, start_date, end_date):
    filtered_df = df[(df['category'] == selected_category) &
                     (df['date'] >= start_date) &
                     (df['date'] <= end_date)]

    product_sales = filtered_df.groupby('product')['sales'].sum().reset_index()
    product_sales = product_sales.sort_values('sales', ascending=False).head(10)

    fig = px.bar(product_sales, x='product', y='sales',
                title=f'Top 10 Products: {selected_category}')
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
```

## 9. Natural Language Processing (NLP)

NLP adalah bidang yang fokus pada interaksi antara komputer dan bahasa manusia.

### Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Alternatively, Lemmatization
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)
```

### Word Embeddings

```python
# Training Word2Vec with Gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

# Assume sentences is a list of tokenized sentences
# sentences = [['I', 'love', 'machine', 'learning'], ['This', 'is', 'great'], ...]

# Detect bigrams
bigram = Phrases(sentences, min_count=5)
bigram_mod = Phraser(bigram)
sentences = [bigram_mod[doc] for doc in sentences]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model.train(sentences, total_examples=len(sentences), epochs=10)

# Save and load model
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# Get word vector
vector = model.wv['machine_learning']

# Find similar words
similar_words = model.wv.most_similar('machine_learning', topn=10)
```

### Text Classification with Transformers

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Prepare data
texts = ["I love this product!", "This is terrible.", ...]
labels = [1, 0, ...]  # 1 for positive, 0 for negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize and encode
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Convert to PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(y_train)
)
test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(y_test)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Fine-tune model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print(f"Epoch {epoch+1}, Accuracy: {correct/total:.4f}")

# Save model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
```

## 10. Computer Vision

Computer Vision adalah bidang AI yang memungkinkan komputer untuk memahami dan menginterpretasikan informasi visual.

### Image Classification dengan Convolutional Neural Networks

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Transfer Learning dengan ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Fine-tuning
# Unfreeze beberapa layer teratas dari ResNet
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile model dengan learning rate yang lebih kecil
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Lanjutkan training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```

### Object Detection dengan YOLO

```python
# Menggunakan YOLOv5 dengan PyTorch
import torch

# Clone YOLOv5 repository
# !git clone https://github.com/ultralytics/yolov5

# Menginstal dependencies
# !pip install -r yolov5/requirements.txt

# Load pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Perform inference
results = model('path/to/image.jpg')

# Visualisasi hasil
results.show()

# Akses deteksi
detections = results.pandas().xyxy[0]  # mengambil bounding boxes
```

### Image Segmentation dengan U-Net

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder (downsampling path)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder (upsampling path)
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create dan train model
model = unet_model()
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
```

## 11. AutoML dan Hyperparameter Optimization

AutoML (Automated Machine Learning) adalah teknologi yang mengotomatisasi proses membangun dan mengoptimalkan model machine learning.

### AutoML dengan Auto-Sklearn

```python
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Load data
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1)

# Configure Auto-Sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,  # 2 menit untuk optimasi
    per_run_time_limit=30,        # 30 detik per run
    tmp_folder='/tmp/autosklearn_temp',
    delete_tmp_folder_after_terminate=True
)

# Fit model
automl.fit(X_train, y_train)

# Get performance
y_pred = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, y_pred))

# Print model details
print(automl.sprint_statistics())
print(automl.show_models())
```

### Hyperparameter Optimization dengan Optuna

```python
import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define objective function
def objective(trial):
    # Parameter space
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )

    # Evaluate
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return score.mean()

# Create study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters
print('Best trial:')
print('  Value: {}'.format(study.best_trial.value))
print('  Params: ')
for key, value in study.best_trial.params.items():
    print('    {}: {}'.format(key, value))

# Visualisasi hasil
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_optimization_history(study)
```

## 12. Reinforcement Learning

Reinforcement Learning adalah jenis machine learning di mana agen belajar untuk mengambil tindakan dalam lingkungan untuk memaksimalkan reward.

### Deep Q-Learning

```python
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Inisialisasi environment dan agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training loop
n_episodes = 1000
batch_size = 32

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(500):  # max 500 timesteps per episode
        # env.render()  # uncomment untuk visualisasi
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Reward modification to penalize failure
        reward = reward if not done else -10

        # Remember experience
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print(f"episode: {e+1}/{n_episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
            break

    # Train the agent with experiences in memory
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
```

## 13. Model Interpretability

Model interpretability adalah kemampuan untuk memahami dan menjelaskan keputusan model machine learning.

### LIME (Local Interpretable Model-agnostic Explanations)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime import lime_tabular

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Explain prediction for a specific instance
i = 0  # instance to explain
exp = explainer.explain_instance(
    X_test[i],
    model.predict_proba,
    num_features=10
)

# Visualize the explanation
exp.show_in_notebook()

# Get explanation as text
explanation = exp.as_list()
```

### SHAP (SHapley Additive exPlanations)

```python
import shap

# Initialize explainer with model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot of feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for a specific instance
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test[0,:], feature_names=feature_names)

# Dependence plot
shap.dependence_plot("feature_index", shap_values[1], X_test, feature_names=feature_names)
```

### Partial Dependence Plots (PDP)

```python
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Create PDP for specific features
features = [0, 1, (0, 1)]  # feature indices to analyze
fig, ax = plt.subplots(figsize=(12, 6))
plot_partial_dependence(
    model, X_train, features,
    feature_names=feature_names,
    grid_resolution=50,
    ax=ax
)

plt.tight_layout()
plt.show()
```

## 14. Ethical AI and Fairness

Membangun model AI yang etis dan adil adalah aspek penting dalam data science modern.

### Mendeteksi Bias dalam Model

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Siapkan dataset dengan protected attribute
df_dummy = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'protected_attribute': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
})
df_dummy['label'] = (df_dummy['feature1'] > 0).astype(int)

# Buat dataset objek
dataset = BinaryLabelDataset(
    df=df_dummy,
    label_names=['label'],
    protected_attribute_names=['protected_attribute']
)

# Hitung metrik fairness
metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{'protected_attribute': 0}],
    privileged_groups=[{'protected_attribute': 1}]
)

# Statistical Parity Difference (SPD)
print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")
# Nilai mendekati 0 menunjukkan keseimbangan

# Mitigasi bias dengan reweighing
reweighing = Reweighing(
    unprivileged_groups=[{'protected_attribute': 0}],
    privileged_groups=[{'protected_attribute': 1}]
)
dataset_transformed = reweighing.fit_transform(dataset)

# Hitung
```

# Panduan Lengkap Aspek Data Science Lanjutan

## 1. Penanganan Data Tidak Seimbang (Imbalanced Data)

Data tidak seimbang adalah masalah umum dalam proyek data science, terutama dalam klasifikasi ketika satu kelas memiliki jumlah sampel yang jauh lebih sedikit dibandingkan kelas lainnya.

### 1.1 Teknik Sampling

#### SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE menciptakan sampel sintetis dari kelas minoritas dengan melakukan interpolasi di antara sampel yang sudah ada.

```python
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Membuat dataset tidak seimbang
X, y = make_classification(n_samples=5000, n_features=10, n_informative=8,
                          n_redundant=2, n_clusters_per_class=1,
                          weights=[0.9, 0.1], random_state=42)

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Terapkan SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Melihat distribusi kelas sebelum dan sesudah SMOTE
print(f"Distribusi kelas sebelum SMOTE: {np.bincount(y_train)}")
print(f"Distribusi kelas setelah SMOTE: {np.bincount(y_train_resampled)}")
```

#### ADASYN (Adaptive Synthetic Sampling)

ADASYN menyerupai SMOTE tetapi berfokus pada sampel yang lebih sulit diidentifikasi, menghasilkan lebih banyak sampel sintetis untuk sampel minoritas yang sulit diklasifikasikan.

```python
from imblearn.over_sampling import ADASYN

# Terapkan ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

print(f"Distribusi kelas setelah ADASYN: {np.bincount(y_train_resampled)}")
```

#### RandomUnderSampler

Teknik ini mengurangi jumlah sampel dari kelas mayoritas secara acak hingga mencapai keseimbangan yang diinginkan.

```python
from imblearn.under_sampling import RandomUnderSampler

# Terapkan Random Under Sampling
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print(f"Distribusi kelas setelah Random Under Sampling: {np.bincount(y_train_resampled)}")
```

#### NearMiss

NearMiss adalah metode under-sampling yang memilih sampel kelas mayoritas berdasarkan jarak ke sampel kelas minoritas.

```python
from imblearn.under_sampling import NearMiss

# Terapkan NearMiss
nm = NearMiss(version=1)
X_train_resampled, y_train_resampled = nm.fit_resample(X_train, y_train)

print(f"Distribusi kelas setelah NearMiss: {np.bincount(y_train_resampled)}")
```

### 1.2 Class Weights

Banyak algoritma machine learning mendukung parameter `class_weight` yang memberi bobot lebih pada kelas minoritas.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Menggunakan class weights pada Random Forest
# 'balanced' secara otomatis menyesuaikan bobot berbanding terbalik dengan frekuensi kelas
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Atau dengan bobot yang ditentukan secara manual
class_weights = {0: 1, 1: 10}  # Kelas 1 diberi bobot 10x lebih tinggi
rf_custom = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
rf_custom.fit(X_train, y_train)

y_pred_custom = rf_custom.predict(X_test)
print(classification_report(y_test, y_pred_custom))
```

### 1.3 Evaluasi Metrik Khusus

Untuk data tidak seimbang, akurasi saja sering menyesatkan. Metrik berikut lebih informatif:

#### F1-Score

F1-score adalah rata-rata harmonik dari precision dan recall.

```python
from sklearn.metrics import f1_score

# Hitung F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.4f}")
```

#### Precision-Recall AUC

PR-AUC lebih informatif daripada ROC-AUC untuk data yang sangat tidak seimbang.

```python
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Menggunakan probabilitas untuk kelas positif
y_proba = rf.predict_proba(X_test)[:, 1]

# Hitung precision dan recall untuk berbagai threshold
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

# Plot kurva Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.', label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```

#### Cohen's Kappa

Cohen's Kappa mengukur kecocokan antara prediksi dan ground truth, dengan mempertimbangkan kemungkinan kecocokan acak.

```python
from sklearn.metrics import cohen_kappa_score

# Hitung Cohen's Kappa
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")
```

### 1.4 Threshold Adjustment

Untuk masalah klasifikasi biner, menyesuaikan threshold keputusan dapat meningkatkan performa pada kelas minoritas.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

# Mendapatkan probabilitas untuk kelas positif
y_proba = rf.predict_proba(X_test)[:, 1]

# Menemukan threshold optimal berdasarkan F1-score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"Threshold default: 0.5")
print(f"Threshold optimal untuk F1-score: {optimal_threshold:.4f}")

# Membuat prediksi dengan threshold optimal
y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)
print("\nHasil dengan threshold default:")
print(classification_report(y_test, y_pred))
print("\nHasil dengan threshold optimal:")
print(classification_report(y_test, y_pred_adjusted))
```

## 2. Statistical Testing dan Hypothesis Testing

Pengujian statistik sangat penting untuk memvalidasi kesimpulan dalam data science dan untuk mendukung pengambilan keputusan berbasis data.

### 2.1 A/B Testing

A/B Testing adalah metode eksperimental untuk membandingkan dua versi dari sesuatu untuk menentukan yang mana yang lebih baik.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulasi data untuk dua variasi
np.random.seed(42)
# Kelompok kontrol (A)
control_mean = 5.0
control_size = 1000
control = np.random.normal(control_mean, 1.0, control_size)

# Kelompok treatment (B) dengan efek positif kecil
treatment_effect = 0.2  # 4% peningkatan
treatment_mean = control_mean * (1 + treatment_effect)
treatment_size = 1000
treatment = np.random.normal(treatment_mean, 1.0, treatment_size)

# Visualisasi distribusi
plt.figure(figsize=(10, 6))
plt.hist(control, alpha=0.5, label='Control (A)')
plt.hist(treatment, alpha=0.5, label='Treatment (B)')
plt.axvline(control_mean, color='blue', linestyle='dashed', linewidth=1)
plt.axvline(treatment_mean, color='orange', linestyle='dashed', linewidth=1)
plt.legend()
plt.title('Distribusi Metrik untuk Kontrol vs Treatment')
plt.show()

# Melakukan t-test untuk menguji perbedaan mean
t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)

print(f"Mean Kontrol: {np.mean(control):.4f}")
print(f"Mean Treatment: {np.mean(treatment):.4f}")
print(f"Perbedaan Relatif: {(np.mean(treatment) - np.mean(control)) / np.mean(control) * 100:.2f}%")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpretasi hasil
alpha = 0.05
if p_value < alpha:
    print(f"Dengan tingkat signifikansi {alpha}, kita menolak hipotesis nol.")
    print("Ada bukti statistik yang signifikan bahwa mean kedua kelompok berbeda.")
else:
    print(f"Dengan tingkat signifikansi {alpha}, kita gagal menolak hipotesis nol.")
    print("Tidak ada bukti statistik yang signifikan bahwa mean kedua kelompok berbeda.")
```

#### Perhitungan Ukuran Sampel untuk A/B Testing

Ukuran sampel yang tepat penting untuk memastikan power statistik yang cukup.

```python
from statsmodels.stats.power import TTestIndPower

# Menghitung ukuran sampel untuk A/B test
effect_size = 0.2  # Effect size yang diinginkan (Cohen's d)
alpha = 0.05       # Tingkat signifikansi
power = 0.8        # Power statistik yang diinginkan (1 - probabilitas Type II error)

# Inisialisasi objek power analysis
analysis = TTestIndPower()

# Menghitung ukuran sampel yang dibutuhkan
sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                  power=power, ratio=1.0, alternative='two-sided')

print(f"Untuk mendeteksi effect size {effect_size:.2f} dengan alpha={alpha} dan power={power}")
print(f"Ukuran sampel yang dibutuhkan per kelompok: {int(np.ceil(sample_size))}")

# Menggambar kurva power
sample_sizes = np.arange(10, 500, 10)
power_curve = analysis.power(effect_size=effect_size, nobs1=sample_sizes,
                            alpha=alpha, ratio=1.0, alternative='two-sided')

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, power_curve, 'b-', linewidth=2)
plt.axhline(y=power, color='r', linestyle='--', label=f'Target Power={power}')
plt.xlabel('Sample Size per Group')
plt.ylabel('Power (1-β)')
plt.title(f'Power Analysis untuk Effect Size {effect_size}')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 Statistical Significance

Signifikansi statistik menunjukkan seberapa kecil kemungkinan hasil terjadi karena kebetulan.

#### p-value dan Confidence Interval

```python
import numpy as np
import scipy.stats as stats

# Contoh: Apakah suatu intervensi meningkatkan skor test?
# H0: Intervensi tidak memiliki efek (μ_diff = 0)
# H1: Intervensi memiliki efek positif (μ_diff > 0)

# Data sebelum intervensi
before = np.array([72, 68, 74, 77, 65, 70, 71, 69, 73, 76])

# Data setelah intervensi
after = np.array([77, 75, 76, 82, 71, 79, 80, 75, 78, 81])

# Paired t-test (karena sampel sebelum dan sesudah dari subyek yang sama)
t_stat, p_value = stats.ttest_rel(after, before, alternative='greater')

# Hitung rata-rata perbedaan dan interval konfidensi
diff = after - before
mean_diff = np.mean(diff)
se = stats.sem(diff)
n = len(diff)
dof = n - 1

# 95% confidence interval
ci_low, ci_high = stats.t.interval(0.95, dof, loc=mean_diff, scale=se)

print(f"Mean perbedaan: {mean_diff:.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"95% Confidence Interval: ({ci_low:.2f}, {ci_high:.2f})")

# Interpretasi
alpha = 0.05
if p_value < alpha:
    print(f"Dengan tingkat signifikansi {alpha}, kita menolak hipotesis nol.")
    print("Ada bukti statistik yang cukup bahwa intervensi meningkatkan skor.")
else:
    print(f"Dengan tingkat signifikansi {alpha}, kita gagal menolak hipotesis nol.")
    print("Tidak ada bukti statistik yang cukup bahwa intervensi meningkatkan skor.")
```

### 2.3 Power Analysis

Power Analysis menentukan kemampuan test untuk mendeteksi efek jika benar-benar ada.

```python
from statsmodels.stats.power import TTestPower

# Menghitung statistical power untuk paired t-test
effect_size = 0.8  # Cohen's d
sample_size = 10   # Ukuran sampel yang sudah ada
alpha = 0.05       # Tingkat signifikansi

# Inisialisasi objek power analysis untuk paired t-test
analysis = TTestPower()
power = analysis.solve_power(effect_size=effect_size, nobs=sample_size,
                           alpha=alpha, alternative='larger')

print(f"Effect Size (Cohen's d): {effect_size:.2f}")
print(f"Ukuran Sampel: {sample_size}")
print(f"Alpha: {alpha}")
print(f"Power: {power:.4f}")

if power < 0.8:
    # Jika power kurang dari standar 0.8, hitung ukuran sampel yang diperlukan
    required_n = analysis.solve_power(effect_size=effect_size, power=0.8,
                                    alpha=alpha, alternative='larger')
    print(f"Power kurang dari 0.8. Ukuran sampel yang disarankan: {int(np.ceil(required_n))}")
```

### 2.4 Multiple Testing Correction

Ketika melakukan banyak pengujian statistik secara simultan, risiko kesalahan Tipe I (false positives) meningkat.

#### Bonferroni Correction

```python
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Simulasi data: 100 uji hipotesis ketika semuanya H0 benar
np.random.seed(42)
p_values = np.random.uniform(0, 1, 100)

# Beberapa p-values buatan yang signifikan
p_values[np.random.choice(100, 10, replace=False)] = np.random.uniform(0, 0.05, 10)

# Terapkan koreksi Bonferroni
alpha = 0.05
reject_bonferroni = p_values < (alpha / len(p_values))

# Terapkan berbagai metode koreksi dengan statsmodels
methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'fdr_bh']
results = pd.DataFrame()

for method in methods:
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    results[f'reject_{method}'] = reject
    results[f'corrected_p_{method}'] = pvals_corrected

# Hitung jumlah penolakan untuk setiap metode
results['p_values'] = p_values
results['significant_uncorrected'] = p_values < alpha

print(f"Total pengujian: {len(p_values)}")
print(f"Signifikan tanpa koreksi: {sum(p_values < alpha)}")

for method in methods:
    print(f"Signifikan setelah koreksi {method}: {sum(results[f'reject_{method}'])}")

# Plot perbandingan p-values sebelum dan sesudah koreksi
plt.figure(figsize=(12, 6))
plt.scatter(range(len(p_values)), np.sort(p_values), label='Original')

for method in methods:
    plt.scatter(range(len(p_values)), np.sort(results[f'corrected_p_{method}']),
               alpha=0.6, label=method)

plt.axhline(y=alpha, color='r', linestyle='--', label=f'Alpha={alpha}')
plt.xlabel('Uji Hipotesis')
plt.ylabel('p-value')
plt.title('P-values Sebelum dan Sesudah Koreksi Multiple Testing')
plt.legend()
plt.show()
```

#### False Discovery Rate (FDR)

FDR mengontrol proporsi false positives di antara semua temuan positif.

```python
# FDR dengan metode Benjamini-Hochberg sudah tercakup dalam contoh di atas ('fdr_bh')
# Analisis lebih detail spesifik untuk FDR

# Membuat data dengan beberapa sinyal nyata
np.random.seed(42)
n_tests = 1000
# 90% hipotesis nol benar, 10% memiliki efek nyata
true_null = np.random.uniform(0, 1, 900)
true_alternative = np.random.beta(1, 10, 100)  # Distribusi yang condong ke p-values kecil
p_values = np.concatenate([true_null, true_alternative])
np.random.shuffle(p_values)

# Hitung q-values (FDR-adjusted p-values)
_, q_values, _, _ = multipletests(p_values, method='fdr_bh')

# Plot distribusi p-values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(p_values, bins=20, alpha=0.7)
plt.axhline(y=n_tests/20, color='r', linestyle='--',
           label='Expected under null hypothesis')
plt.title('Distribusi p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(q_values, bins=20, alpha=0.7)
plt.axvline(x=0.05, color='r', linestyle='--', label='q-value = 0.05')
plt.title('Distribusi q-values (FDR-adjusted)')
plt.xlabel('q-value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Hitung FDR pada berbagai threshold
thresholds = np.arange(0.01, 0.2, 0.01)
results_fdr = []

for threshold in thresholds:
    # Discoveries (rejected null hypotheses)
    discoveries = sum(p_values < threshold)
    # Expected false discoveries under uniform distribution (null hypotheses)
    expected_false_discoveries = threshold * n_tests
    # Estimated FDR
    est_fdr = expected_false_discoveries / max(discoveries, 1)
    results_fdr.append((threshold, discoveries, expected_false_discoveries, est_fdr))

df_fdr = pd.DataFrame(results_fdr,
                    columns=['Threshold', 'Discoveries', 'Expected False Discoveries', 'Estimated FDR'])
print(df_fdr)
```

## 3. Causal Inference

Causal inference berusaha mengidentifikasi hubungan sebab-akibat, bukan sekadar korelasi.

### 3.1 Propensity Score Matching (PSM)

PSM adalah teknik untuk mengurangi bias seleksi dalam studi observasional.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Simulasi dataset
np.random.seed(42)
n = 1000

# Kovariat
age = np.random.normal(50, 10, n)
income = np.random.normal(50000, 15000, n)
health = np.random.normal(70, 15, n)

# Kecenderungan untuk treatment (dipengaruhi oleh kovariat)
logit = -2 + 0.05 * age + 0.00002 * income - 0.02 * health
p_treatment = 1 / (1 + np.exp(-logit))
treatment = np.random.binomial(1, p_treatment)

# Outcome: dipengaruhi oleh kovariat dan treatment
outcome = 50 + 0.5 * age - 0.0001 * income + 0.2 * health + 10 * treatment + np.random.normal(0, 10, n)

# Buat dataframe
data = pd.DataFrame({
    'age': age,
    'income': income,
    'health': health,
    'treatment': treatment,
    'outcome': outcome
})

# Estimasi propensity scores
covariates = ['age', 'income', 'health']
X = data[covariates]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model propensity score
ps_model = LogisticRegression()
ps_model.fit(X_scaled, data['treatment'])
data['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]

# Visualisasi distribusi propensity score
plt.figure(figsize=(10, 6))
plt.hist(data.loc[data['treatment']==1, 'propensity_score'], alpha=0.5, bins=20, label='Treatment')
plt.hist(data.loc[data['treatment']==0, 'propensity_score'], alpha=0.5, bins=20, label='Control')
plt.title('Distribusi Propensity Scores')
plt.xlabel('Propensity Score')
plt.ylabel('Count')
plt.legend()
plt.show()

# Matching menggunakan Nearest Neighbor
# Pisahkan antara kelompok treatment dan control
treated = data[data['treatment'] == 1]
control = data[data['treatment'] == 0]

# Inisialisasi Nearest Neighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])

# Cari match untuk setiap individu di kelompok treatment
distances, indices = nn.kneighbors(treated[['propensity_score']])

# Dapatkan matched pairs
matched_control_indices = indices.flatten()
matched_control = control.iloc[matched_control_indices].copy()
matched_control['matched_to'] = treated.index.values

# Kombinasikan data
matched_data = pd.concat([treated, matched_control])

# Evaluasi keseimbangan kovariat sebelum dan sesudah matching
def evaluate_balance(data_before, data_after, covariates, treatment_col):
    balance_before = []
    balance_after = []

    for covariate in covariates:
        # Sebelum matching
        treated_mean_before = data_before[data_before[treatment_col]==1][covariate].mean()
        control_mean_before = data_before[data_before[treatment_col]==0][covariate].mean()
        std_before = data_before[covariate].std()
        std_diff_before = (treated_mean_before - control_mean_before) / std_before

        # Setelah matching
        treated_after = data_after[data_after[treatment_col]==1]
        control_after = data_after[data_after[treatment_col]==0]
        treated_mean_after = treated_after[covariate].mean()
        control_mean_after = control_after[covariate].mean()
        std_diff_after = (treated_mean_after - control_mean_after) / std_before

        balance_before.append(std_diff_before)
        balance_after.append(std_diff_after)

        print(f"{covariate}: SMD before = {std_diff_before:.4f}, SMD after = {std_diff_after:.4f}")

    return balance_before, balance_after

print("Standardized Mean Differences (SMD) Before and After Matching:")
balance_before, balance_after = evaluate_balance(data, matched_data, covariates, 'treatment')

# Plot balance sebelum dan sesudah matching
plt.figure(figsize=(10, 6))
plt.axvline(x=0, color='black', linestyle='-')
plt.axvline(x=-0.1, color='gray', linestyle='--')
plt.axvline(x=0.1, color='gray', linestyle='--')
plt.plot(balance_before, range(len(covariates)), 'o', label='Before Matching')
plt.plot(balance_after, range(len(covariates)), 's', label='After Matching')
plt.yticks(range(len(covariates)), covariates)
plt.xlabel('Standardized Mean Difference')
plt.title('Covariate Balance Before and After Matching')
plt.legend()
plt.grid(True)
plt.show()

# Estimasi average treatment effect
ate_naive = data[data['treatment']==1]['outcome'].mean() - data[data['treatment']==0]['outcome'].mean()
ate_matched = matched_data[matched_data['treatment']==1]['outcome'].mean() - matched_data[matched_data['treatment']==0]['outcome'].mean()

print(f"Naive ATE estimate: {ate_naive:.4f}")
print(f"PSM ATE estimate: {ate_matched:.4f}")
print(f"True treatment effect in simulation: 10.0000")
```

### 3.2 Difference-in-Differences (DiD)

DiD adalah metode quasi-eksperimental yang membandingkan perubahan outcome antara kelompok treatment dan kontrol.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Simulasi data DiD
np.random.seed(42)
n_units = 100
n_time = 10
treatment_time = 5  # Intervensi terjadi setelah t=5

# Generate panel data
unit_ids = np.repeat(np.arange(n_units), n_time)
time_ids = np.tile(np.arange(n_time), n_units)
treatment_group = np.repeat(np.random.binomial(1, 0.5, n_units), n_time)
post_treatment = (time_ids >= treatment_time).astype(int)
did = treatment_group * post_treatment

# Generate outcomes dengan fixed effects dan treatment effect
unit_effects = np.repeat(np.random.normal(0, 2, n_units), n_time)
time_effects = np.tile(np.arange(n_time) * 0.5, n_units)  # Trend waktu linear
treatment_effect = 2.5  # Efek treatment yang sebenarnya
error = np.random.normal(0, 1, n_units * n_time)

# Outcome: Unit fixed effects + time trend + treatment effect + error
outcome = unit_effects + time_effects + did * treatment_effect + error

# Buat DataFrame
data = pd.DataFrame({
    'unit': unit_ids,
    'time': time_ids,
    'treatment_group': treatment_group,
    'post': post_treatment,
    'did': did,
    'outcome': outcome
})

# Plot trend untuk treatment dan control group
mean_by_group_time = data.groupby(['treatment_group', 'time'])['outcome'].mean().reset_index()
treatment_means = mean_by_group_time[mean_by_group_time['treatment_group'] == 1]
control_means = mean_by_group_time[mean_by_group_time['treatment_group'] == 0]

plt.figure(figsize=(10, 6))
plt.plot(treatment_means['time'], treatment_means['outcome'], 'b-', marker='o', label='Treatment Group')
plt.plot(control_means['time'], control_means['outcome'], 'r-', marker='s', label='Control Group')
plt.axvline(x=treatment_time-0.5, color='gray', linestyle='--', label='Treatment Time')
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.title('Difference-in-Differences: Treatment vs Control Group Trends')
plt.legend()
plt.grid(True)
plt.show()

# DiD Model
model = smf.ols('outcome ~ treatment_group + post + did', data=data).fit()
print(model.summary().tables[1])

# Model dengan fixed effects
model_fe = smf.ols('outcome ~ treatment_group + post + did + C(unit) + C(time)', data=data).fit()
print("Treatment effect estimate from fixed effects model:")
print(model_fe.summary().tables[1].data[4])

# Parallel trends assumption check
# Interaksi antara kelompok treatment dan periode waktu sebelum treatment
pre_treatment_data = data[data['time'] < treatment_time].copy()
pre_treatment_data['time_cat'] = pre_treatment_data['time'].astype('category')
model_parallel = smf.ols('outcome ~ C(time_cat) + treatment_group + C(time_cat):treatment_group',
                        data=pre_treatment_data).fit()

print("\nTest for parallel trends (interaction terms should not be significant):")
for i in range(treatment_time):
    if i > 0:  # Skip base period
        var_name = f"C(time_cat)[{i}]:treatment_group"
        coef = model_parallel.params.get(var_name, None)
        p_val = model_parallel.pvalues.get(var_name, None)
        if coef is not None and p_val is not None:
            print(f"Time {i} × Treatment: coef = {coef:.4f}, p-value = {p_val:.4f}")
```

### 3.3 Instrumental Variables (IV)

IV adalah teknik untuk mengestimasi efek kausal ketika ada endogenitas (variabel penjelas berkorelasi dengan error).

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Simulasi data dengan endogenitas
np.random.seed(42)
n = 10000

# Instrumen (eksogen)
z = np.random.normal(0, 1, n)

# Unobserved confounding
u = np.random.normal(0, 1, n)

# Treatment (endogen, dipengaruhi oleh instrumen dan confounding)
x = 0.5 * z + 0.5 * u + np.random.normal(0, 1, n)

# Outcome (dipengaruhi oleh treatment dan confounding)
true_effect = 2.0  # Efek kausal yang sebenarnya
y = true_effect * x + u + np.random.normal(0, 1, n)

# Buat DataFrame
data = pd.DataFrame({
    'y': y,
    'x': x,
    'z': z
})

# OLS - akan menghasilkan estimasi yang bias karena endogenitas
X_ols = sm.add_constant(data['x'])
model_ols = sm.OLS(data['y'], X_ols).fit()

# IV estimation dengan 2SLS
# Stage 1: Regresikan X pada Z
X_stage1 = sm.add_constant(data['z'])
model_stage1 = sm.OLS(data['x'], X_stage1).fit()
x_hat = model_stage1.predict()

# Stage 2: Regresikan Y pada X_hat
X_stage2 = sm.add_constant(x_hat)
model_stage2 = sm.OLS(data['y'], X_stage2).fit()

# 2SLS dengan linearmodels package
formula = 'y ~ 1 + [x ~ z]'
model_iv = IV2SLS.from_formula(formula, data=data).fit(cov_type='robust')

print("OLS Estimate (Biased):")
print(f"Estimated effect: {model_ols.params['x']:.4f}")
print(f"Standard error: {model_ols.bse['x']:.4f}")
print(f"p-value: {model_ols.pvalues['x']:.4f}")

print("\nManual 2SLS Estimate:")
print(f"Estimated effect: {model_stage2.params[1]:.4f}")
print(f"Standard error: {model_stage2.bse[1]:.4f}")
print(f"p-value: {model_stage2.pvalues[1]:.4f}")

print("\nIV2SLS Estimate:")
print(model_iv.summary.tables[1])

print(f"True causal effect: {true_effect:.1f}")

# Test instrumen
# Stage 1 regression - instrumen harus kuat (F-statistic > 10)
print("\nFirst Stage Regression:")
print(f"F-statistic: {model_stage1.fvalue:.4f}")
print(f"p-value: {model_stage1.f_pvalue:.4f}")
print(f"R-squared: {model_stage1.rsquared:.4f}")
```

### 3.4 Causal Graphical Models

Menggunakan directed acyclic graphs (DAGs) untuk memodelkan hubungan kausal.

```python
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pydot
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Membuat DAG sederhana
G = nx.DiGraph()

# Menambahkan nodes dan edges
G.add_nodes_from(['Education', 'Income', 'Job', 'Health', 'Exercise'])
G.add_edges_from([
    ('Education', 'Job'),
    ('Education', 'Income'),
    ('Job', 'Income'),
    ('Job', 'Exercise'),
    ('Income', 'Health'),
    ('Exercise', 'Health')
])

# Plot DAG
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
       node_size=2000, arrowsize=20, font_size=12, font_weight='bold')
plt.title('Causal Directed Acyclic Graph (DAG)')
plt.show()

# Implementasi Bayesian Network dengan pgmpy
model = BayesianNetwork([
    ('Education', 'Job'),
    ('Education', 'Income'),
    ('Job', 'Income'),
    ('Job', 'Exercise'),
    ('Income', 'Health'),
    ('Exercise', 'Health')
])

# Contoh CPDs (untuk simplifikasi semua variabel biner)
# CPD untuk Education (root node)
cpd_education = TabularCPD(
    variable='Education', variable_card=2,
    values=[[0.7], [0.3]],  # [Low, High]
    state_names={'Education': ['Low', 'High']}
)

# CPD untuk Job kondisional pada Education
cpd_job = TabularCPD(
    variable='Job', variable_card=2,
    values=[[0.8, 0.3],
           [0.2, 0.7]],  # [Low skill, High skill] given [Low, High] Education
    evidence=['Education'],
    evidence_card=[2],
    state_names={'Job': ['Low_skill', 'High_skill'], 'Education': ['Low', 'High']}
)

# CPD untuk Income kondisional pada Education dan Job
cpd_income = TabularCPD(
    variable='Income', variable_card=2,
    values=[[0.9, 0.7, 0.6, 0.1],
           [0.1, 0.3, 0.4, 0.9]],  # [Low, High] Income given combinations of Education and Job
    evidence=['Education', 'Job'],
    evidence_card=[2, 2],
    state_names={'Income': ['Low', 'High'], 'Education': ['Low', 'High'], 'Job': ['Low_skill', 'High_skill']}
)

# CPD untuk Exercise kondisional pada Job
cpd_exercise = TabularCPD(
    variable='Exercise', variable_card=2,
    values=[[0.7, 0.4],
           [0.3, 0.6]],  # [Low, High] Exercise given Job
    evidence=['Job'],
    evidence_card=[2],
    state_names={'Exercise': ['Low', 'High'], 'Job': ['Low_skill', 'High_skill']}
)

# CPD untuk Health kondisional pada Income dan Exercise
cpd_health = TabularCPD(
    variable='Health', variable_card=2,
    values=[[0.8, 0.6, 0.5, 0.1],
           [0.2, 0.4, 0.5, 0.9]],  # [Poor, Good] Health given combinations of Income and Exercise
    evidence=['Income', 'Exercise'],
    evidence_card=[2, 2],
    state_names={'Health': ['Poor', 'Good'], 'Income': ['Low', 'High'], 'Exercise': ['Low', 'High']}
)

# Tambahkan CPDs ke model
model.add_cpds(cpd_education, cpd_job, cpd_income, cpd_exercise, cpd_health)

# Verifikasi model
assert model.check_model()

# Inferensi
inference = VariableElimination(model)

# Simulasi intervensi causal (do-operator): P(Health | do(Exercise = High))
# Ini berbeda dari kondisional P(Health | Exercise = High) karena menghilangkan backdoor paths

# 1. Kondisional biasa P(Health | Exercise = High)
result_conditional = inference.query(variables=['Health'], evidence={'Exercise': 'High'})
print("P(Health | Exercise = High):")
print(result_conditional)

# 2. Intervensi do(Exercise = High)
# Dalam kasus ini, intervensi memutus edge dari Job ke Exercise
# Kita perlu membuat model baru untuk menghitung ini
model_do_exercise = BayesianNetwork([
    ('Education', 'Job'),
    ('Education', 'Income'),
    ('Job', 'Income'),
    # Edge dari Job ke Exercise dihapus
    ('Income', 'Health'),
    ('Exercise', 'Health')
])

# Update CPDs
cpd_exercise_do = TabularCPD(
    variable='Exercise', variable_card=2,
    values=[[0], [1]],  # Selalu High karena do(Exercise = High)
    state_names={'Exercise': ['Low', 'High']}
)

model_do_exercise.add_cpds(cpd_education, cpd_job, cpd_income, cpd_exercise_do, cpd_health)
assert model_do_exercise.check_model()

inference_do = VariableElimination(model_do_exercise)
result_do = inference_do.query(variables=['Health'])
print("\nP(Health | do(Exercise = High)):")
print(result_do)

print("\nMembandingkan efek kondisional vs kausal:")
print(f"P(Health = Good | Exercise = High) = {result_conditional.values[1]:.4f}")
print(f"P(Health = Good | do(Exercise = High)) = {result_do.values[1]:.4f}")
```

### 3.5 Rubin Causal Model

Framework untuk inferensi kausal berdasarkan potential outcomes.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulasi data potential outcomes
np.random.seed(42)
n = 1000

# Kovariat (X)
age = np.random.normal(50, 10, n)
income = np.random.normal(50000, 15000, n)
health_score = np.random.normal(70, 15, n)

# Potential outcomes
# Y(0): outcome jika tidak treatment
# Y(1): outcome jika treatment
noise = np.random.normal(0, 5, n)
Y0 = 50 + 0.3 * age - 0.0001 * income + 0.2 * health_score + noise
ate = 10  # Average treatment effect
hte = 0.1 * health_score  # Heterogeneous treatment effect based on health
Y1 = Y0 + ate + hte + np.random.normal(0, 2, n)

# Peluang mendapatkan treatment (propensity score)
logit = -3 + 0.02 * age + 0.00001 * income - 0.01 * health_score
p_treatment = 1 / (1 + np.exp(-logit))
treatment = np.random.binomial(1, p_treatment)

# Observed outcome (kita hanya mengobservasi salah satu dari Y0 atau Y1)
Y_obs = treatment * Y1 + (1 - treatment) * Y0

# Dataset observasional
data = pd.DataFrame({
    'age': age,
    'income': income,
    'health_score': health_score,
    'treatment': treatment,
    'outcome': Y_obs,
    # Berikut ini tidak terobservasi dalam dunia nyata, hanya untuk evaluasi
    'Y0': Y0,
    'Y1': Y1,
    'true_effect': Y1 - Y0,
    'propensity': p_treatment
})

# Hitung true average treatment effect (ATE)
true_ate = np.mean(Y1 - Y0)
print(f"True ATE: {true_ate:.4f}")

# Naive estimator (bias karena confounding)
naive_ate = data[data['treatment']==1]['outcome'].mean() - data[data['treatment']==0]['outcome'].mean()
print(f"Naive ATE estimate: {naive_ate:.4f}")
print(f"Naive estimator bias: {naive_ate - true_ate:.4f}")

# Inverse Probability Weighting (IPW)
# Mengestimasi propensity score
from sklearn.linear_model import LogisticRegression

X = data[['age', 'income', 'health_score']]
T = data['treatment']
y = data['outcome']

# Model propensity score
ps_model = LogisticRegression()
ps_model.fit(X, T)
data['estimated_ps'] = ps_model.predict_proba(X)[:, 1]

# IPW estimator
data['ipw'] = treatment / data['estimated_ps'] + (1 - treatment) / (1 - data['estimated_ps'])
data['weighted_outcome'] = data['outcome'] * data['ipw']

ipw_ate = np.mean(data[data['treatment'] == 1]['weighted_outcome']) - np.mean(data[data['treatment'] == 0]['weighted_outcome'])
print(f"IPW ATE estimate: {ipw_ate:.4f}")
print(f"IPW estimator bias: {ipw_ate - true_ate:.4f}")

# Doubly Robust Estimator
# 1. Model outcome
from sklearn.linear_model import LinearRegression

# Outcome model for control group
model_control = LinearRegression()
model_control.fit(X[T == 0], y[T == 0])
data['predicted_Y0'] = model_control.predict(X)

# Outcome model for treatment group
model_treated = LinearRegression()
model_treated.fit(X[T == 1], y[T == 1])
data['predicted_Y1'] = model_treated.predict(X)

# 2. Doubly robust estimator
data['dr_term'] = (treatment * (data['outcome'] - data['predicted_Y1']) / data['estimated_ps']) - \
                  ((1 - treatment) * (data['outcome'] - data['predicted_Y0']) / (1 - data['estimated_ps'])) + \
                  (data['predicted_Y1'] - data['predicted_Y0'])

dr_ate = np.mean(data['dr_term'])
print(f"Doubly Robust ATE estimate: {dr_ate:.4f}")
print(f"Doubly Robust estimator bias: {dr_ate - true_ate:.4f}")

# Visualisasi heterogeneous treatment effects
plt.figure(figsize=(10, 6))
plt.scatter(data['health_score'], data['true_effect'], alpha=0.5)
plt.xlabel('Health Score')
plt.ylabel('Individual Treatment Effect')
plt.title('Heterogeneous Treatment Effects by Health Score')
plt.grid(True)
plt.show()

# Estimasi Conditional Average Treatment Effect (CATE)
# Misalnya, CATE berdasarkan health_score
bins = pd.qcut(data['health_score'], 5)
cate_by_health = data.groupby(bins)['true_effect'].mean()

plt.figure(figsize=(10, 6))
cate_by_health.plot(kind='bar', yerr=data.groupby(bins)['true_effect'].std() / np.sqrt(data.groupby(bins).size()))
plt.xlabel('Health Score Quintile')
plt.ylabel('Conditional Average Treatment Effect')
plt.title('CATE by Health Score Quintile')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
```

## 4. Anomaly Detection

Anomaly detection mengidentifikasi data points, events, atau observations yang menyimpang dari pola normal dalam dataset.

### 4.1 Isolation Forest

Isolation Forest secara efisien mendeteksi outliers dengan mengisolasi observasi.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Membuat dataset dengan outliers
np.random.seed(42)

# Data normal
X_inliers, _ = make_blobs(n_samples=990, centers=3, n_features=2, random_state=42)

# Outliers
X_outliers = np.random.uniform(low=-6, high=6, size=(10, 2))

# Gabungkan data
X = np.vstack([X_inliers, X_outliers])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Terapkan Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
y_pred = clf.fit_predict(X_scaled)
scores = clf.decision_function(X_scaled)

# Konversi output ke binary (1: inlier, 0: outlier) dan ke anomaly score
is_inlier = y_pred == 1
anomaly_score = -scores  # Higher values mean more anomalous

# Plot hasil
plt.figure(figsize=(10, 8))

# Plot semua titik data
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=anomaly_score, cmap='viridis',
           alpha=0.7, edgecolors='k', s=50)

# Tandai outliers terdeteksi
outliers = X_scaled[~is_inlier]
plt.scatter(outliers[:, 0], outliers[:, 1], color='red',
           marker='x', s=150, linewidth=3, label='Detected Outliers')

plt.colorbar(label='Anomaly Score')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Evaluasi performa
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# True labels: terakhir 10 data adalah outliers
y_true = np.ones(X.shape[0])
y_true[-10:] = -1  # Convention: -1 for outliers, 1 for inliers

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"Precision: {precision_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"F1-score: {f1_score(y_true, y_pred, pos_label=-1):.4f}")

# Parameter tuning dan analisis
contamination_rates = [0.005, 0.01, 0.02, 0.05, 0.1]
results = []

for c in contamination_rates:
    clf = IsolationForest(contamination=c, random_state=42)
    y_pred = clf.fit_predict(X_scaled)

    precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)

    results.append({
        'contamination': c,
        'n_outliers_detected': sum(y_pred == -1),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

pd.DataFrame(results).set_index('contamination')
```

### 4.2 Local Outlier Factor (LOF)

LOF mengidentifikasi outliers berdasarkan kepadatan lokal.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_moons

# Membuat dataset
np.random.seed(42)

# Data normal (bentuk bulan sabit)
X_inliers, _ = make_moons(n_samples=990, noise=0.1, random_state=42)

# Outliers acak
X_outliers = np.random.uniform(low=-1, high=2, size=(10, 2))

# Gabungkan data
X = np.vstack([X_inliers, X_outliers])

# Terapkan LOF
n_neighbors = 20
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.01)
y_pred = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_  # Higher values indicate more anomalous

# Konversi output
is_inlier = y_pred == 1

# Plot hasil
plt.figure(figsize=(10, 8))

# Plot semua data
plt.scatter(X[:, 0], X[:, 1], c=lof_scores, cmap='YlOrRd',
           alpha=0.7, edgecolors='k', s=50)

# Highlight outliers terdeteksi
outliers = X[~is_inlier]
plt.scatter(outliers[:, 0], outliers[:, 1], color='red',
           marker='x', s=150, linewidth=3, label='Detected Outliers')

plt.colorbar(label='LOF Score (Higher is more anomalous)')
plt.title(f'Local Outlier Factor (LOF) Anomaly Detection (n_neighbors={n_neighbors})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Evaluasi performa
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# True labels: terakhir 10 data adalah outliers
y_true = np.ones(X.shape[0])
y_true[-10:] = -1  # Convention: -1 for outliers, 1 for inliers

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"Precision: {precision_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"F1-score: {f1_score(y_true, y_pred, pos_label=-1):.4f}")

# Analisis efek jumlah tetangga
neighbors_range = [5, 10, 20, 50, 100]
results = []

for n in neighbors_range:
    lof = LocalOutlierFactor(n_neighbors=n, contamination=0.01)
    y_pred = lof.fit_predict(X)

    precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)

    results.append({
        'n_neighbors': n,
        'n_outliers_detected': sum(y_pred == -1),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

pd.DataFrame(results).set_index('n_neighbors')
```

### 4.3 One-Class SVM

One-Class SVM mencari hyperplane yang memisahkan data dari origin dengan margin maksimum.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Membuat dataset dengan outliers
np.random.seed(42)

# Data normal
X_inliers, _ = make_blobs(n_samples=990, centers=1, n_features=2, random_state=42)

# Outliers
X_outliers = np.random.uniform(low=-10, high=10, size=(10, 2))

# Gabungkan data
X = np.vstack([X_inliers, X_outliers])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Terapkan One-Class SVM
nu = 0.01  # Expected proportion of outliers
svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
y_pred = svm.fit_predict(X_scaled)
scores = svm.decision_function(X_scaled)

# Konversi output (1: inlier, -1: outlier)
is_inlier = y_pred == 1
anomaly_score = -scores  # Higher values mean more anomalous

# Plot hasil
plt.figure(figsize=(10, 8))

# Plot data points
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=anomaly_score, cmap='viridis',
           alpha=0.7, edgecolors='k', s=50)

# Highlight outliers
outliers = X_scaled[~is_inlier]
plt.scatter(outliers[:, 0], outliers[:, 1], color='red',
           marker='x', s=150, linewidth=3, label='Detected Outliers')

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='dashed')

plt.colorbar(label='Anomaly Score')
plt.title('One-Class SVM Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Evaluasi performa
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# True labels: terakhir 10 data adalah outliers
y_true = np.ones(X.shape[0])
y_true[-10:] = -1  # Convention: -1 for outliers, 1 for inliers

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"Precision: {precision_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"F1-score: {f1_score(y_true, y_pred, pos_label=-1):.4f}")

# Parameter tuning
nu_values = [0.005, 0.01, 0.02, 0.05, 0.1]
gamma_values = ['scale', 'auto', 0.01, 0.1, 1.0]

results = []

for nu_val in nu_values:
    for gamma_val in gamma_values:
        svm = OneClassSVM(nu=nu_val, kernel='rbf', gamma=gamma_val)
        y_pred = svm.fit_predict(X_scaled)

        precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=-1)
        f1 = f1_score(y_true, y_pred, pos_label=-1)

        results.append({
            'nu': nu_val,
            'gamma': gamma_val,
            'n_outliers_detected': sum(y_pred == -1),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

pd.DataFrame(results)
```

# Panduan Lengkap Data Science: Dari Anomaly Detection hingga Uncertainty Quantification

## 16. Anomaly Detection

### Autoencoders untuk Anomaly Detection

Autoencoder adalah jenis neural network yang dilatih untuk merekonstruksi input aslinya. Dalam konteks deteksi anomali, autoencoder dilatih menggunakan data normal, sehingga diharapkan dapat merekonstruksi data normal dengan baik tetapi gagal merekonstruksi anomali. Kesalahan rekonstruksi yang tinggi mengindikasikan anomali.

**Prinsip Kerja:**

1. **Encoder**: Memetakan input ke ruang laten berdimensi lebih rendah
2. **Decoder**: Merekonstruksi input dari representasi laten
3. **Loss Function**: Mengukur perbedaan antara input asli dan hasil rekonstruksi

**Implementasi dengan Keras:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Mempersiapkan data
# Asumsikan X adalah dataset normal yang sudah diproses
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Membangun model autoencoder
input_dim = X_train.shape[1]
encoding_dim = 10  # Dimensi ruang laten

# Input layer
input_layer = Input(shape=(input_dim,))

# Encoder
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
encoder = Dense(encoding_dim, activation="relu")(encoder)

# Decoder
decoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(decoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

# Model lengkap
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Melatih autoencoder hanya dengan data normal
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test))

# Menghitung rekonstruksi untuk data uji
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Menentukan threshold untuk deteksi anomali
threshold = np.percentile(mse, 95)  # Misalnya 95% persentil

# Fungsi deteksi anomali
def detect_anomalies(data):
    data_scaled = scaler.transform(data)
    reconstructions = autoencoder.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    return mse > threshold
```

**Kelebihan:**

-   Dapat menangkap pola kompleks dan non-linear
-   Bekerja baik dengan data berdimensi tinggi
-   Tidak memerlukan label anomali untuk pelatihan

**Kekurangan:**

-   Membutuhkan jumlah data normal yang cukup besar
-   Sensitif terhadap hyperparameter dan arsitektur
-   Performa dapat menurun jika terdapat anomali dalam data pelatihan

### Time Series Anomaly Detection

Deteksi anomali pada data deret waktu memerlukan pendekatan khusus karena adanya aspek temporal. Teknik ini mengidentifikasi titik atau segmen yang menyimpang secara signifikan dari pola deret waktu normal.

**Teknik Utama:**

1. **Statistical Methods**:

    - Moving Average dan Standard Deviation
    - Seasonal Decomposition (STL)
    - ARIMA-based Methods

2. **Machine Learning Methods**:

    - Autoencoders untuk deret waktu
    - LSTM Autoencoders
    - Isolation Forest dengan fitur temporal

**Implementasi LSTM Autoencoder:**

```python
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# Asumsikan X_train_ts adalah data deret waktu dengan bentuk (samples, timesteps, features)
# Mengubah format data untuk time series
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Parameter
seq_length = 30  # Window size
X_sequences = create_sequences(X_scaled, seq_length)
X_train_seq, X_test_seq = train_test_split(X_sequences, test_size=0.2, random_state=42)

# Membangun LSTM Autoencoder
timesteps = X_train_seq.shape[1]
n_features = X_train_seq.shape[2]

# Model
input_layer = Input(shape=(timesteps, n_features))
encoded = LSTM(32, activation='relu')(input_layer)
encoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(n_features))(decoded)

lstm_autoencoder = Model(inputs=input_layer, outputs=decoded)
lstm_autoencoder.compile(optimizer='adam', loss='mse')

# Melatih model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
history = lstm_autoencoder.fit(
    X_train_seq, X_train_seq,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_seq, X_test_seq),
    callbacks=[early_stopping],
    shuffle=False
)

# Menghitung error rekonstruksi
X_test_pred = lstm_autoencoder.predict(X_test_seq)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test_seq), axis=(1, 2))

# Menentukan threshold
threshold = np.max(test_mae_loss)  # Atau metode lain untuk menentukan threshold

# Deteksi anomali pada data baru
def detect_anomalies_ts(new_data, threshold):
    new_sequences = create_sequences(scaler.transform(new_data), seq_length)
    preds = lstm_autoencoder.predict(new_sequences)
    mae_loss = np.mean(np.abs(preds - new_sequences), axis=(1, 2))
    anomalies = mae_loss > threshold
    return anomalies, mae_loss
```

**Aplikasi Anomaly Detection:**

-   Deteksi fraud pada transaksi keuangan
-   Pemantauan kesehatan mesin (predictive maintenance)
-   Keamanan jaringan dan deteksi intrusi
-   Pemantauan kualitas dalam manufaktur
-   Deteksi outlier pada data pasien

## 17. Recommender Systems

Sistem rekomendasi adalah algoritma yang dirancang untuk memprediksi preferensi pengguna dan memberikan saran yang sesuai dengan minat atau kebutuhan mereka.

### Collaborative Filtering

Collaborative filtering merekomendasikan item kepada pengguna berdasarkan preferensi pengguna lain yang memiliki pola perilaku serupa.

**User-Based Collaborative Filtering:**
Menemukan pengguna serupa dan merekomendasikan item yang disukai oleh pengguna-pengguna tersebut.

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Contoh data ratings dalam format (user_id, item_id, rating)
# Membuat user-item matrix
user_item_matrix = pd.pivot_table(ratings_df,
                                  values='rating',
                                  index='user_id',
                                  columns='item_id',
                                  fill_value=0)

# Menghitung kesamaan antar pengguna
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)

# Fungsi untuk mendapatkan rekomendasi
def get_user_recommendations(user_id, top_n=5):
    # Mendapatkan item yang belum dilihat oleh pengguna
    user_ratings = user_item_matrix.loc[user_id]
    unseen_items = user_ratings[user_ratings == 0].index

    # Pengguna yang mirip dengan pengguna target
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]

    # Menghitung prediksi rating
    recommendations = {}
    for item in unseen_items:
        pred_rating = 0
        similarity_sum = 0

        for sim_user, similarity in similar_users.items():
            # Jika pengguna serupa telah memberikan rating pada item
            if user_item_matrix.loc[sim_user, item] > 0:
                pred_rating += similarity * user_item_matrix.loc[sim_user, item]
                similarity_sum += similarity

        if similarity_sum > 0:
            recommendations[item] = pred_rating / similarity_sum

    # Mendapatkan top-n rekomendasi
    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_recommendations
```

**Item-Based Collaborative Filtering:**
Menemukan item serupa dengan item yang disukai pengguna.

```python
# Menghitung kesamaan antar item
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=user_item_matrix.columns,
                                  columns=user_item_matrix.columns)

# Fungsi untuk mendapatkan rekomendasi berbasis item
def get_item_recommendations(user_id, top_n=5):
    # Item yang sudah dilihat dan belum dilihat
    user_ratings = user_item_matrix.loc[user_id]
    seen_items = user_ratings[user_ratings > 0].index
    unseen_items = user_ratings[user_ratings == 0].index

    # Menghitung prediksi rating
    recommendations = {}

    for unseen in unseen_items:
        pred_rating = 0
        similarity_sum = 0

        for seen in seen_items:
            similarity = item_similarity_df.loc[unseen, seen]
            # Jika terdapat kesamaan yang cukup
            if similarity > 0.1:  # Threshold untuk kesamaan
                pred_rating += similarity * user_ratings[seen]
                similarity_sum += similarity

        if similarity_sum > 0:
            recommendations[unseen] = pred_rating / similarity_sum

    # Mendapatkan top-n rekomendasi
    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_recommendations
```

### Content-Based Filtering

Content-based filtering merekomendasikan item berdasarkan kemiripan fitur atau konten item dengan preferensi pengguna di masa lalu.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Asumsi data: items_df memiliki kolom 'description' berisi teks deskriptif tentang item
# Membuat TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items_df['description'])

# Menghitung kesamaan kosinus antar item
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi berdasarkan konten
def get_content_recommendations(item_id, top_n=5):
    # Mendapatkan indeks item
    idx = items_df[items_df['item_id'] == item_id].index[0]

    # Menghitung skor kesamaan dengan semua item
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Mengurutkan item berdasarkan skor kesamaan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mendapatkan top-n item serupa (tidak termasuk item itu sendiri)
    sim_scores = sim_scores[1:top_n+1]

    # Mendapatkan indeks item
    item_indices = [i[0] for i in sim_scores]

    # Mengembalikan item yang direkomendasikan
    return items_df.iloc[item_indices]
```

### Matrix Factorization

Matrix Factorization memecah matriks user-item menjadi dua matriks faktor laten yang mewakili pengguna dan item dalam ruang berdimensi rendah.

```python
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Asumsi ratings_df memiliki kolom 'user_id', 'item_id', 'rating'
# Menyiapkan data untuk surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)

# Membagi data
trainset, testset = train_test_split(data, test_size=0.2)

# Membangun dan melatih model
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Mendapatkan rekomendasi untuk pengguna
def get_mf_recommendations(user_id, items_to_predict, top_n=5):
    # Prediksi rating untuk item yang belum dilihat
    predictions = []
    for item_id in items_to_predict:
        predicted_rating = model.predict(user_id, item_id).est
        predictions.append((item_id, predicted_rating))

    # Mengurutkan berdasarkan rating prediksi
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Mengembalikan top-n rekomendasi
    return predictions[:top_n]
```

### Deep Learning untuk Rekomendasi

Deep learning dapat digunakan untuk membangun sistem rekomendasi yang lebih kompleks dan menangkap pola non-linear.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Asumsi: n_users, n_items adalah jumlah pengguna dan item
# embedding_size adalah dimensi ruang laten

# Input layers
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# Embedding layers
user_embedding = Embedding(n_users, embedding_size, name='user_embedding')(user_input)
item_embedding = Embedding(n_items, embedding_size, name='item_embedding')(item_input)

# Flatten embeddings
user_vector = Flatten(name='user_vector')(user_embedding)
item_vector = Flatten(name='item_vector')(item_embedding)

# Concatenate user and item vectors
concat = Concatenate()([user_vector, item_vector])

# Dense layers
dense1 = Dense(128, activation='relu')(concat)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(1)(dense2)

# Create model
model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Melatih model
# X_train harus dalam format [user_indices, item_indices]
# y_train adalah rating aktual
history = model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1
)

# Fungsi untuk mendapatkan rekomendasi
def get_dl_recommendations(user_id, items_to_predict, top_n=5):
    # Membuat input untuk model
    user_array = np.array([user_id] * len(items_to_predict))
    item_array = np.array(items_to_predict)

    # Prediksi ratings
    predictions = model.predict([user_array, item_array]).flatten()

    # Mengurutkan berdasarkan rating prediksi
    item_predictions = list(zip(items_to_predict, predictions))
    item_predictions.sort(key=lambda x: x[1], reverse=True)

    # Mengembalikan top-n rekomendasi
    return item_predictions[:top_n]
```

### Hybrid Methods

Metode hybrid menggabungkan berbagai pendekatan untuk menutupi kekurangan masing-masing metode dan memberikan rekomendasi yang lebih akurat.

```python
# Fungsi untuk menggabungkan hasil rekomendasi dari beberapa model
def hybrid_recommendations(user_id, items_to_predict, top_n=5):
    # Mendapatkan rekomendasi dari setiap model
    cf_recs = get_user_recommendations(user_id)
    cb_recs = get_content_recommendations(user_favorite_item)
    mf_recs = get_mf_recommendations(user_id, items_to_predict)

    # Menggabungkan rekomendasi dengan pembobotan
    cf_weight = 0.4
    cb_weight = 0.3
    mf_weight = 0.3

    # Dictionary untuk menyimpan skor gabungan
    hybrid_scores = {}

    # Mengumpulkan item dan skor dari collaborative filtering
    for item_id, score in cf_recs:
        if item_id not in hybrid_scores:
            hybrid_scores[item_id] = 0
        hybrid_scores[item_id] += score * cf_weight

    # Mengumpulkan item dan skor dari content-based filtering
    for _, row in cb_recs.iterrows():
        item_id = row['item_id']
        if item_id not in hybrid_scores:
            hybrid_scores[item_id] = 0
        # Asumsikan ada skor kesamaan dalam row
        hybrid_scores[item_id] += row['similarity'] * cb_weight

    # Mengumpulkan item dan skor dari matrix factorization
    for item_id, score in mf_recs:
        if item_id not in hybrid_scores:
            hybrid_scores[item_id] = 0
        hybrid_scores[item_id] += score * mf_weight

    # Mengurutkan item berdasarkan skor gabungan
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    # Mengembalikan top-n rekomendasi
    return sorted_items[:top_n]
```

**Aplikasi Sistem Rekomendasi:**

-   E-commerce (produk)
-   Platform streaming (film, musik, video)
-   Media sosial (konten, koneksi)
-   Berita dan artikel
-   Lowongan kerja

## 18. Graph/Network Analysis

Analisis jaringan/graf memungkinkan kita memahami hubungan dan interaksi antara entitas dalam sistem yang kompleks.

### Node Embedding

Node embedding adalah teknik yang memetakan node dalam graf ke dalam ruang vektor berdimensi rendah, sehingga node yang serupa dalam graf memiliki representasi vektor yang dekat.

```python
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# Membuat graf contoh
G = nx.fast_gnp_random_graph(100, 0.05)

# Inisialisasi model Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Melatih model Node2Vec
model = node2vec.fit(window=10, min_count=1)

# Mendapatkan embedding untuk node tertentu
node_id = 0
node_embedding = model.wv.get_vector(str(node_id))

# Mencari node yang serupa
similar_nodes = model.wv.most_similar(str(node_id))
print(f"Node yang serupa dengan node {node_id}: {similar_nodes}")

# Visualisasi embedding dengan t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Mendapatkan embedding untuk semua node
embeddings = np.zeros((G.number_of_nodes(), 64))
for i in range(G.number_of_nodes()):
    embeddings[i] = model.wv.get_vector(str(i))

# Mengurangi dimensi dengan t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
node_tsne = tsne.fit_transform(embeddings)

# Visualisasi
plt.figure(figsize=(10, 8))
plt.scatter(node_tsne[:, 0], node_tsne[:, 1], alpha=0.7)
plt.title('Visualisasi Node Embedding dengan t-SNE')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')
plt.show()
```

### Graph Neural Networks (GNN)

GNN adalah model deep learning yang dirancang khusus untuk data graf, yang dapat mengekstrak fitur dari struktur graf dan atribut node.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Membuat model Graph Convolutional Network (GCN)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Contoh penggunaan
# Asumsikan kita memiliki adjacency matrix dan fitur node
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Daftar edge
x = torch.randn((3, 16))  # 3 node dengan 16 fitur masing-masing
y = torch.tensor([0, 1, 0], dtype=torch.long)  # Label node

# Membuat objek Data untuk PyTorch Geometric
data = Data(x=x, edge_index=edge_index, y=y)

# Inisialisasi model
model = GCN(input_dim=16, hidden_dim=32, output_dim=2)  # 2 kelas

# Definisikan optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluasi
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred == data.y).sum().item()
acc = correct / data.y.size(0)
print(f'Accuracy: {acc:.4f}')
```

### Community Detection

Community detection mengidentifikasi kelompok node yang terhubung erat dalam jaringan, sehingga menunjukkan struktur modular dari jaringan tersebut.

```python
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

# Membuat graf dengan struktur komunitas
G = nx.karate_club_graph()  # Graf klub karate Zachary yang terkenal

# Deteksi komunitas menggunakan algoritma Louvain
partition = community_louvain.best_partition(G)

# Menghitung jumlah komunitas
communities = set(partition.values())
print(f"Jumlah komunitas terdeteksi: {len(communities)}")

# Visualisasi komunitas
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
cmap = plt.cm.tab10
colors = [cmap(i) for i in partition.values()]

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=200, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title('Komunitas dalam Graf Klub Karate Zachary')
plt.axis('off')
plt.show()

# Analisis komunitas
for community_id in communities:
    members = [node for node, community in partition.items() if community == community_id]
    print(f"Komunitas {community_id}: {members}")
```

### Link Prediction

Link prediction memprediksi kemungkinan terbentuknya edge atau hubungan antara node yang belum terhubung dalam graf.

```python
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

# Membuat graf contoh
G = nx.fast_gnp_random_graph(100, 0.1)

# Membagi edge menjadi training dan testing
edges = list(G.edges())
edges_to_remove = train_test_split(edges, test_size=0.2)[1]

# Membuat graf training dengan menghapus edges testing
G_train = G.copy()
G_train.remove_edges_from(edges_to_remove)

# Fungsi untuk menghitung berbagai metrik kedekatan node
def calculate_features(G, node_pair):
    u, v = node_pair

    # Common neighbors
    common_neighbors = list(nx.common_neighbors(G, u, v))
    n_common_neighbors = len(common_neighbors)

    # Jaccard coefficient
    jaccard = 0
    try:
        jaccard = len(common_neighbors) / (len(set(G.neighbors(u))) + len(set(G.neighbors(v))) - len(common_neighbors))
    except ZeroDivisionError:
        pass

    # Preferential attachment
    preferential = len(list(G.neighbors(u))) * len(list(G.neighbors(v)))

    # Adamic-Adar index
    adamic_adar = 0
    for w in common_neighbors:
        try:
            adamic_adar += 1 / np.log(len(list(G.neighbors(w))))
        except:
            pass

    return [n_common_neighbors, jaccard, preferential, adamic_adar]

# Menyiapkan data untuk link prediction
# Positive examples: edges yang dihapus
positive_examples = edges_to_remove

# Negative examples: node pairs yang tidak memiliki edge
nodes = list(G.nodes())
all_possible_edges = [(u, v) for u in nodes for v in nodes if u < v]  # Hindari duplikasi
negative_examples = [edge for edge in all_possible_edges if edge not in edges]
negative_examples = np.random.choice(
    negative_examples,
    size=len(positive_examples),
    replace=False
).tolist()

# Menggabungkan positive dan negative examples
X_pairs = positive_examples + negative_examples
y = [1] * len(positive_examples) + [0] * len(negative_examples)

# Ekstrak fitur
X = [calculate_features(G_train, pair) for pair in X_pairs]

# Membangun model prediksi link
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluasi model
from sklearn.metrics import accuracy_score, precision_score, recall_score

y_pred = model.predict(X)
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall: {recall_score(y, y_pred):.4f}")
```

### Centrality Measures

Ukuran sentralitas menentukan pentingnya node dalam jaringan berdasarkan posisi dan konektivitasnya.

```python
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Membuat graf contoh
G = nx.barabasi_albert_graph(50, 3)

# Menghitung berbagai ukuran sentralitas
centrality_measures = {
    'Degree': nx.degree_centrality(G),
    'Betweenness': nx.betweenness_centrality(G),
    'Closeness': nx.closeness_centrality(G),
    'Eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
    'PageRank': nx.pagerank(G, alpha=0.85)
}

# Membuat DataFrame untuk analisis
centrality_df = pd.DataFrame(centrality_measures)

# Melihat korelasi antar ukuran sentralitas
correlation = centrality_df.corr()
print("Korelasi antar ukuran sentralitas:")
print(correlation)

# Visualisasi graf dengan node size berdasarkan sentralitas
plt.figure(figsize=(15, 10))

# Membuat subplot untuk setiap ukuran sentralitas
for i, (measure, centrality) in enumerate(centrality_measures.items()):
    plt.subplot(2, 3, i+1)

    # Posisi node
    pos = nx.spring_layout(G, seed=42)

    # Ukuran node berdasarkan sentralitas
    node_size = [v * 5000 for v in centrality.values()]

    nx.draw_networkx(G, pos, node_size=node_size, with_labels=False,
                    node_color='lightblue', edge_color='gray', alpha=0.8)

    plt.title(f'{measure} Centrality')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Mengidentifikasi node paling penting berdasarkan sentralitas
top_nodes = {}
for measure, centrality in centrality_measures.items():
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes[measure] = sorted_nodes[:5]  # 5 node teratas

print("Top 5 node berdasarkan setiap ukuran sentralitas:")
for measure, nodes in top_nodes.items():
    print(f"{measure}: {nodes}")
```

**Aplikasi Analisis Jaringan:**

-   Analisis jaringan sosial
-   Deteksi komunitas dalam jaringan kompleks
-   Identifikasi influencer dalam pemasaran
-   Analisis jaringan komunikasi
-   Analisis jaringan protein dalam bioinformatika
-   Analisis jaringan transportasi dan logistik

## 19. Advanced Evaluation Techniques

Teknik evaluasi lanjutan membantu kita memahami dan meningkatkan performa model machine learning secara lebih mendalam.

### Bayesian Optimization

Bayesian Optimization adalah teknik untuk mengoptimalkan fungsi yang mahal untuk dievaluasi (seperti hyperparameter model ML) dengan menggunakan model probabilistik.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# Membuat dataset klasifikasi
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

# Mendefinisikan model
rf = RandomForestClassifier(random_state=42)

# Mendefinisikan ruang parameter
param_space = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

# Bayesian Optimization dengan cross-validation
opt = BayesSearchCV(
    rf,
    param_space,
    n_iter=50,  # Jumlah iterasi
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    random_state=42,
    verbose=1
)

# Menjalankan optimasi
opt.fit(X, y)

# Melihat hasil
print(f"Best parameters: {opt.best_params_}")
print(f"Best cross-validation score: {opt.best_score_:.4f}")

# Membuat model dengan parameter terbaik
best_rf = RandomForestClassifier(**opt.best_params_, random_state=42)
best_rf.fit(X, y)
```

### Learning Curves

Learning curve menunjukkan performa model sebagai fungsi dari ukuran dataset pelatihan, yang membantu mendiagnosis underfitting atau overfitting.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

# Memuat dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Fungsi untuk membuat learning curve
def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy'
    )

    # Menghitung rata-rata dan standar deviasi
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

    # Plot standar deviasi
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid(True)

    return plt

# Membuat learning curve untuk SVM
svm = SVC(kernel='rbf', gamma='scale')
plot_learning_curve(svm, X, y)
plt.show()
```

### Validation Curves

Validation curve menunjukkan performa model sebagai fungsi dari nilai hyperparameter tertentu, yang membantu dalam pemilihan hyperparameter optimal.

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

# Mendefinisikan model
dt = DecisionTreeClassifier(random_state=42)

# Mendefinisikan range hyperparameter yang akan dievaluasi
param_range = np.arange(1, 21)

# Menghitung validation curve
train_scores, test_scores = validation_curve(
    dt, X, y, param_name='max_depth', param_range=param_range,
    cv=5, scoring='accuracy'
)

# Menghitung rata-rata dan standar deviasi
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
plt.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')

# Plot standar deviasi
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

plt.title('Validation Curve')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

### Bias-Variance Tradeoff Analysis

Analisis bias-variance membantu memahami kesalahan prediksi model dari perspektif bias (kesalahan karena asumsi yang terlalu sederhana) dan variance (sensitifitas terhadap fluktuasi dalam data pelatihan).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Membuat data sintetis
np.random.seed(42)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fungsi untuk menghitung bias dan variance
def bias_variance_decomposition(degrees, X_train, X_test, y_train, y_test, n_bootstraps=200):
    # Inisialisasi array untuk menyimpan hasil
    test_mse = np.zeros(len(degrees))
    bias_squared = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    # Membuat grid untuk visualisasi
    X_grid = np.linspace(0, 1, 1000).reshape(-1, 1)

    plt.figure(figsize=(15, 10))

    for i, degree in enumerate(degrees):
        y_pred_bootstraps = np.zeros((n_bootstraps, len(X_grid)))

        # Membuat model polynomial regression
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

        # Bootstrap sampling
        for j in range(n_bootstraps):
            # Sampling dengan pengembalian
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot, y_boot = X_train[indices], y_train[indices]

            # Melatih model
            model.fit(X_boot, y_boot)

            # Prediksi
            y_pred_bootstraps[j] = model.predict(X_grid)

        # Menghitung prediksi rata-rata
        y_pred_mean = np.mean(y_pred_bootstraps, axis=0)

        # Menghitung true function (dalam kasus ini sin(2πx))
        y_true = np.sin(2 * np.pi * X_grid).ravel()

        # Menghitung bias^2
        bias_squared[i] = np.mean((y_pred_mean - y_true) ** 2)

        # Menghitung variance
        variance[i] = np.mean([np.mean((y_pred_bootstraps[j] - y_pred_mean) ** 2) for j in range(n_bootstraps)])

        # Menghitung MSE pada test set
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_mse[i] = np.mean((y_test_pred - y_test) ** 2)

        # Plot untuk visualisasi
        plt.subplot(2, 3, i+1)
        plt.scatter(X_train, y_train, color='blue', s=10, label='Training data')
        plt.plot(X_grid, y_true, color='green', label='True function')
        plt.plot(X_grid, y_pred_mean, color='red', label='Mean prediction')

        # Plot beberapa model bootstrap
        for j in range(min(10, n_bootstraps)):
            plt.plot(X_grid, y_pred_bootstraps[j], color='gray', alpha=0.1)

        plt.title(f'Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')

        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot bias-variance tradeoff
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, bias_squared, 'o-', color='blue', label='Bias^2')
    plt.plot(degrees, variance, 'o-', color='orange', label='Variance')
    plt.plot(degrees, bias_squared + variance, 'o-', color='red', label='Total Error')
    plt.plot(degrees, test_mse, 'o-', color='green', label='Test MSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.show()

    return bias_squared, variance, test_mse

# Menjalankan analisis bias-variance
degrees = [1, 3, 5, 7, 9]
bias_squared, variance, test_mse = bias_variance_decomposition(degrees, X_train, X_test, y_train, y_test)

# Menampilkan hasil
results = pd.DataFrame({
    'Degree': degrees,
    'Bias^2': bias_squared,
    'Variance': variance,
    'Total Error': bias_squared + variance,
    'Test MSE': test_mse
})
print(results)
```

### Permutation Importance

Permutation importance mengukur seberapa banyak performa model menurun ketika suatu fitur diacak, yang memberikan gambaran tentang pentingnya fitur tersebut.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Memuat dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Menghitung permutation importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

# Mengurutkan fitur berdasarkan importance
sorted_idx = result.importances_mean.argsort()

# Visualisasi
plt.figure(figsize=(12, 8))
plt.barh(range(X.shape[1]), result.importances_mean[sorted_idx])
plt.yticks(range(X.shape[1]), [feature_names[i] for i in sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance (Permutation)')
plt.tight_layout()
plt.show()

# Membandingkan dengan Feature Importance dari RandomForest
plt.figure(figsize=(12, 8))
sorted_idx_rf = rf.feature_importances_.argsort()
plt.barh(range(X.shape[1]), rf.feature_importances_[sorted_idx_rf])
plt.yticks(range(X.shape[1]), [feature_names[i] for i in sorted_idx_rf])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (RandomForest)')
plt.tight_layout()
plt.show()
```

## 20. Distributed Computing untuk Data Science

Komputasi terdistribusi memungkinkan pemrosesan data skala besar yang tidak dapat ditangani oleh satu mesin.

### Hadoop Ecosystem

Hadoop adalah framework open-source untuk penyimpanan dan pemrosesan data terdistribusi.

```python
# Contoh PySpark untuk memproses data dengan Hadoop
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc

# Inisialisasi Spark Session
spark = SparkSession.builder \
    .appName("DistributedDataAnalysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Membaca data dari HDFS
df = spark.read.csv("hdfs:///path/to/data.csv", header=True, inferSchema=True)

# Melakukan transformasi dan agregasi
result = df.groupBy("category") \
    .agg(count("*").alias("count")) \
    .orderBy(desc("count"))

# Menampilkan hasil
result.show()

# Menyimpan hasil ke HDFS
result.write.csv("hdfs:///path/to/output", header=True, mode="overwrite")

# Menutup Spark Session
spark.stop()
```

### Dask/Ray

Dask dan Ray adalah framework Python untuk komputasi paralel dan terdistribusi.

```python
import dask.dataframe as dd
import dask.array as da
import numpy as np
from dask.distributed import Client

# Inisialisasi Dask Client
client = Client()  # Untuk komputasi lokal
# Atau
# client = Client('scheduler-address:8786')  # Untuk cluster

# Membuat Dask DataFrame dari file CSV besar
df = dd.read_csv('big_data_file.csv')

# Melakukan transformasi dan agregasi
result = df.groupby('category').agg({'value': ['mean', 'std', 'count']})

# Komputasi hasil
result = result.compute()

# Operasi dengan Dask Array
# Membuat array besar
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = da.random.random((10000, 10000), chunks=(1000, 1000))

# Operasi matrix
z = da.matmul(x, y)

# Mengeksekusi komputasi
result = z.compute()
```

### GPU Acceleration

Percepatan GPU menggunakan kartu grafis untuk komputasi paralel yang cepat, terutama berguna untuk deep learning.

```python
import tensorflow as tf
import numpy as np

# Memeriksa ketersediaan GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Membuat model neural network dengan TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Mengkompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Membuat data dummy
x_train = np.random.random((1000, 784))
y_train = np.random.randint(0, 10, (1000,))

# Melatih model (otomatis menggunakan GPU jika tersedia)
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

### Optimasi untuk Big Data

Optimasi untuk data berukuran besar fokus pada efisiensi memori dan waktu komputasi.

```python
import pandas as pd
import numpy as np
import dask.dataframe as dd
from joblib import Parallel, delayed

# 1. Optimasi tipe data
def optimize_dtypes(df):
    # Mengubah integer menjadi tipe data yang lebih kecil jika memungkinkan
    int_columns = df.select_dtypes(include=['int']).columns
    for col in int_columns:
        col_min = df[col].min()
        col_max = df[col].max()

        # Menentukan tipe data optimal
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype(np.int32)

    # Mengubah float menjadi tipe data yang lebih kecil
    float_columns = df.select_dtypes(include=['float']).columns
    for col in float_columns:
        df[col] = df[col].astype(np.float32)  # Dari float64 ke float32

    # Mengubah object menjadi category jika jumlah nilai unik sedikit
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        if df[col].nunique() / len(df) < 0.5:  # Jika < 50% nilai unik
            df[col] = df[col].astype('category')

    return df

# 2. Chunking untuk memproses file besar
def process_large_file(file_path, chunk_size=10000):
    # Inisialisasi list untuk menyimpan hasil
    results = []

    # Memproses file dalam chunk
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Melakukan pra-pemrosesan
        chunk = optimize_dtypes(chunk)

        # Melakukan operasi pada chunk
        result = chunk.groupby('category').agg({'value': 'mean'})

        # Menyimpan hasil
        results.append(result)

    # Menggabungkan hasil
    final_result = pd.concat(results)
    return final_result.groupby(final_result.index).mean()

# 3. Pemrosesan paralel dengan joblib
def process_file_parallel(file_paths, n_jobs=-1):
    # Melakukan pemrosesan paralel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_large_file)(file_path) for file_path in file_paths
    )

    # Menggabungkan hasil
    final_result = pd.concat(results)
    return final_result.groupby(final_result.index).mean()

# 4. Menggunakan Dask untuk komputasi terdistribusi
def process_with_dask(file_paths, output_path):
    # Membuat Dask DataFrame
    ddf = dd.read_csv(file_paths)

    # Melakukan operasi
    result = ddf.groupby('category').agg({'value': 'mean'})

    # Menyimpan hasil
    result.compute().to_csv(output_path)
```

## 21. Model Monitoring dan Maintenance

Pemantauan dan pemeliharaan model memastikan performa model tetap optimal seiring berjalannya waktu.

### Data Drift Detection

Data drift terjadi ketika distribusi data input berubah secara signifikan, yang dapat memengaruhi performa model.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest

# Fungsi untuk mendeteksi data drift dengan Kolmogorov-Smirnov test
def detect_drift_ks(reference_data, new_data, threshold=0.05):
    drift_results = {}

    # Mendeteksi drift untuk setiap fitur
    for column in reference_data.columns:
        # Menjalankan KS test
        ks_stat, p_value = ks_2samp(reference_data[column], new_data[column])

        # Jika p-value < threshold, terdapat drift
        drift_detected = p_value < threshold

        drift_results[column] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': drift_detected
        }

    return drift_results

# Fungsi untuk visualisasi distribusi
def plot_distribution_comparison(reference_data, new_data, columns, figsize=(15, 10)):
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    plt.figure(figsize=figsize)

    for i, column in enumerate(columns):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot distribusi reference data
        plt.hist(reference_data[column], bins=30, alpha=0.5, label='Reference')

        # Plot distribusi new data
        plt.hist(new_data[column], bins=30, alpha=0.5, label='New')

        plt.title(f'Distribution of {column}')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Fungsi untuk mendeteksi drift dengan Isolation Forest
def detect_drift_isolation_forest(reference_data, new_data, contamination=0.05):
    # Melatih model dengan reference data
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(reference_data)

    # Prediksi pada new data
    predictions = clf.predict(new_data)
    anomaly_score = clf.decision_function(new_data)

    # Menghitung persentase outlier
    outlier_percentage = (predictions == -1).mean()

    return {
        'outlier_percentage': outlier_percentage,
        'anomaly_scores': anomaly_score,
        'predictions': predictions
    }

# Contoh penggunaan
# Asumsikan reference_data dan new_data adalah DataFrame

# Deteksi drift dengan KS test
drift_results = detect_drift_ks(reference_data, new_data)

# Melihat hasil
drift_columns = [col for col, result in drift_results.items() if result['drift_detected']]
print(f"Fitur dengan drift terdeteksi: {drift_columns}")

# Visualisasi distribusi untuk fitur dengan drift
plot_distribution_comparison(reference_data, new_data, drift_columns)

# Deteksi drift dengan Isolation Forest
if_results = detect_drift_isolation_forest(reference_data, new_data)
print(f"Persentase outlier: {if_results['outlier_percentage'] * 100:.2f}%")
```

# Data Science - Model Monitoring dan Teknik Lanjutan

## 21. Model Monitoring dan Maintenance

### Model Drift Detection

Model drift adalah fenomena ketika performa model machine learning mengalami penurunan seiring waktu. Hal ini terjadi karena distribusi data di dunia nyata berubah dibandingkan dengan data yang digunakan untuk melatih model. Model drift dibagi menjadi dua jenis utama:

#### 1. Concept Drift

Concept drift terjadi ketika hubungan antara fitur input (X) dan variabel target (y) berubah. Misalnya, pada model prediksi harga rumah, faktor yang mempengaruhi harga (seperti lokasi atau ukuran) mungkin berubah signifikansinya seiring waktu.

#### 2. Data Drift (Covariate Shift)

Data drift terjadi ketika distribusi fitur input (X) berubah, meskipun hubungan antara input dan output tetap sama. Contohnya, jika model dilatih dengan data dari satu kota, kemudian diterapkan di kota lain dengan karakteristik berbeda.

#### Teknik Deteksi Model Drift

##### Metode Statistik

```python
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt

# Contoh fungsi untuk mendeteksi data drift menggunakan Kolmogorov-Smirnov test
def deteksi_data_drift(data_referensi, data_baru, kolom, alpha=0.05):
    hasil = {}
    for fitur in kolom:
        statistik, p_value = ks_2samp(data_referensi[fitur], data_baru[fitur])
        drift_terdeteksi = p_value < alpha
        hasil[fitur] = {
            'statistik': statistik,
            'p_value': p_value,
            'drift_terdeteksi': drift_terdeteksi
        }
    return pd.DataFrame(hasil).T

# Visualisasi distribusi fitur
def visualisasi_distribusi(data_referensi, data_baru, fitur):
    plt.figure(figsize=(10, 6))
    plt.hist(data_referensi[fitur], alpha=0.5, label='Data Referensi')
    plt.hist(data_baru[fitur], alpha=0.5, label='Data Baru')
    plt.legend()
    plt.title(f'Perbandingan Distribusi untuk {fitur}')
    plt.xlabel(fitur)
    plt.ylabel('Frekuensi')
    plt.show()
```

##### Monitoring Performa Model

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def monitor_model_performance(y_true, y_pred, periode='mingguan'):
    """
    Menghitung metrik performa model dan memeriksa tren penurunan
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

    # Dalam implementasi nyata, metrik ini akan disimpan untuk analisis tren
    print(f"Metrik performa model ({periode}):")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return metrics
```

##### Teknik Population Stability Index (PSI)

```python
def hitung_psi(data_referensi, data_baru, fitur, bins=10):
    """
    Menghitung Population Stability Index (PSI)
    PSI < 0.1: tidak ada perubahan signifikan
    0.1 < PSI < 0.2: perubahan kecil
    PSI > 0.2: perubahan signifikan
    """
    # Tentukan bin edges berdasarkan data referensi
    bin_edges = np.histogram_bin_edges(data_referensi[fitur], bins=bins)

    # Hitung distribusi data referensi
    actual_count, _ = np.histogram(data_referensi[fitur], bins=bin_edges)
    actual_pct = actual_count / float(len(data_referensi))

    # Hitung distribusi data baru
    expected_count, _ = np.histogram(data_baru[fitur], bins=bin_edges)
    expected_pct = expected_count / float(len(data_baru))

    # Hindari pembagian dengan nol
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)

    # Hitung PSI
    psi_value = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))

    # Interpretasi nilai PSI
    if psi_value < 0.1:
        interpretasi = "Tidak ada perubahan signifikan"
    elif psi_value < 0.2:
        interpretasi = "Perubahan kecil hingga menengah"
    else:
        interpretasi = "Perubahan signifikan terdeteksi"

    return psi_value, interpretasi
```

### A/B Testing untuk Model

A/B Testing adalah metode untuk membandingkan dua versi model untuk menentukan mana yang memberikan performa lebih baik. Dalam konteks machine learning, A/B testing memungkinkan kita untuk mengevaluasi perubahan model sebelum diimplementasikan secara menyeluruh.

```python
import numpy as np
from scipy import stats

def ab_test_model(metrik_model_a, metrik_model_b):
    """
    Melakukan A/B testing untuk membandingkan performa dua model
    """
    # Contoh: Uji t-test untuk memeriksa apakah perbedaan signifikan secara statistik
    t_stat, p_value = stats.ttest_ind(metrik_model_a, metrik_model_b)

    alpha = 0.05
    if p_value < alpha:
        kesimpulan = "Perbedaan signifikan secara statistik"
    else:
        kesimpulan = "Perbedaan tidak signifikan secara statistik"

    perubahan_persen = ((np.mean(metrik_model_b) - np.mean(metrik_model_a)) / np.mean(metrik_model_a)) * 100

    hasil = {
        't_statistic': t_stat,
        'p_value': p_value,
        'perubahan_persen': perubahan_persen,
        'kesimpulan': kesimpulan
    }

    return hasil
```

### Automated Retraining

Automated retraining (pelatihan ulang otomatis) adalah proses mengupdate model secara berkala atau berdasarkan trigger tertentu untuk memastikan model tetap akurat. Berikut adalah implementasi sederhana:

```python
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class AutomatedRetrainer:
    def __init__(self, model, jadwal_days=30, metric_threshold=0.05):
        self.model = model
        self.jadwal_days = jadwal_days
        self.metric_threshold = metric_threshold
        self.last_training_time = time.time()
        self.baseline_performance = None

    def perlu_retraining_berdasarkan_jadwal(self):
        """Memeriksa apakah model perlu dilatih ulang berdasarkan jadwal"""
        waktu_sekarang = time.time()
        selisih_hari = (waktu_sekarang - self.last_training_time) / (60*60*24)
        return selisih_hari >= self.jadwal_days

    def perlu_retraining_berdasarkan_performa(self, current_performance):
        """Memeriksa apakah model perlu dilatih ulang berdasarkan penurunan performa"""
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            return False

        penurunan = (self.baseline_performance - current_performance) / self.baseline_performance
        return penurunan > self.metric_threshold

    def retrain(self, X_train, y_train, X_val=None, y_val=None):
        """Melatih ulang model dengan data baru"""
        print("Melatih ulang model...")

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        self.model.fit(X_train, y_train)
        score = self.model.score(X_val, y_val)

        self.baseline_performance = score
        self.last_training_time = time.time()

        print(f"Model dilatih ulang. Performa baru: {score:.4f}")
        return score
```

### Performance Degradation

Performance degradation (degradasi performa) terjadi ketika model kehilangan akurasi seiring waktu. Berikut adalah beberapa teknik untuk mendeteksi dan mengatasi degradasi performa:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelPerformanceTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.performance_history = pd.DataFrame(
            columns=['timestamp', 'accuracy', 'precision', 'recall', 'f1_score']
        )

    def log_performance(self, y_true, y_pred):
        """Mencatat metrik performa model pada titik waktu tertentu"""
        metrics = {
            'timestamp': pd.Timestamp.now(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        self.performance_history = pd.concat([
            self.performance_history,
            pd.DataFrame([metrics])
        ], ignore_index=True)

        return metrics

    def deteksi_degradasi(self, window_size=5, threshold=0.05):
        """Mendeteksi degradasi performa berdasarkan tren dalam window tertentu"""
        if len(self.performance_history) < window_size + 1:
            return False, "Data tidak cukup untuk deteksi degradasi"

        df = self.performance_history.copy()
        df['rolling_avg_f1'] = df['f1_score'].rolling(window=window_size).mean()

        # Bandingkan rata-rata bergerak terbaru dengan nilai maksimum sebelumnya
        current_avg = df['rolling_avg_f1'].iloc[-1]
        max_avg = df['rolling_avg_f1'].iloc[:-1].max()

        penurunan_relatif = (max_avg - current_avg) / max_avg if max_avg > 0 else 0

        if penurunan_relatif > threshold:
            return True, f"Degradasi terdeteksi: penurunan {penurunan_relatif:.2%}"
        else:
            return False, "Tidak ada degradasi signifikan"

    def visualisasi_tren(self):
        """Memvisualisasikan tren performa model"""
        if len(self.performance_history) < 2:
            print("Data tidak cukup untuk visualisasi tren")
            return

        plt.figure(figsize=(12, 8))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        for metric in metrics:
            plt.plot(
                self.performance_history['timestamp'],
                self.performance_history[metric],
                marker='o',
                label=metric
            )

        plt.title(f'Tren Performa Model {self.model_name}')
        plt.xlabel('Waktu')
        plt.ylabel('Nilai Metrik')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
```

## 22. Domain-Specific Applications

### Financial Time Series Analysis

Analisis deret waktu keuangan adalah penerapan teknik statistik dan machine learning untuk data keuangan time series seperti harga saham, indeks pasar, dan data ekonomi lainnya.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def analisis_saham(symbol, periode='1y'):
    """
    Mengambil dan menganalisis data saham
    """
    # Ambil data dari Yahoo Finance
    data = yf.download(symbol, period=periode)

    # Hitung return harian
    data['Return'] = data['Adj Close'].pct_change() * 100

    # Hitung volatilitas (standar deviasi rolling 21 hari)
    data['Volatilitas'] = data['Return'].rolling(window=21).std()

    # Hitung SMA (Simple Moving Average) 50 dan 200 hari
    data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
    data['SMA200'] = data['Adj Close'].rolling(window=200).mean()

    # Identifikasi sinyal Golden Cross dan Death Cross
    data['Signal'] = 0
    data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1  # Golden Cross (Bullish)
    data.loc[data['SMA50'] < data['SMA200'], 'Signal'] = -1 # Death Cross (Bearish)

    # Visualisasi
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(data['Adj Close'], label='Harga Penutupan')
    plt.plot(data['SMA50'], label='SMA 50 hari')
    plt.plot(data['SMA200'], label='SMA 200 hari')
    plt.title(f'Analisis Harga Saham {symbol}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(data['Return'], label='Return Harian', alpha=0.5)
    plt.plot(data['Volatilitas'], label='Volatilitas (21 hari)', color='red')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Return dan Volatilitas')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return data

def prediksi_harga_saham(data, steps=30):
    """
    Prediksi harga saham menggunakan model ARIMA
    """
    # Gunakan auto_arima untuk menemukan parameter optimal
    model = auto_arima(
        data['Adj Close'],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    # Ringkasan model
    print(model.summary())

    # Forecast
    forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)

    # Buat indeks untuk forecast
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    # Visualisasi
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Adj Close'], label='Historical')
    plt.plot(forecast_dates, forecast, label='Forecast', color='red')

    # Plot confidence interval
    plt.fill_between(
        forecast_dates,
        conf_int[:, 0],
        conf_int[:, 1],
        color='pink',
        alpha=0.3
    )

    plt.title(f'Prediksi Harga Saham untuk {steps} hari ke depan')
    plt.legend()
    plt.show()

    return forecast, conf_int, forecast_dates
```

### Healthcare Data Analytics

Analitik data kesehatan melibatkan penggunaan data pasien, klaim asuransi, dan catatan medis untuk meningkatkan hasil perawatan kesehatan, mengurangi biaya, dan mengoptimalkan alokasi sumber daya.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def segmentasi_pasien(data):
    """
    Melakukan segmentasi pasien berdasarkan karakteristik klinis
    """
    # Pilih fitur untuk clustering
    features = ['Umur', 'Tekanan_Darah_Sistolik', 'Tekanan_Darah_Diastolik',
                'Glukosa_Darah', 'BMI', 'Jumlah_Kunjungan_12Bulan']

    # Normalisasi data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    # Tentukan jumlah cluster optimal dengan metode Elbow
    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # Plot Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Berdasarkan plot, pilih jumlah cluster optimal (misalnya 4)
    optimal_k = 4  # Bisa disesuaikan berdasarkan metode elbow

    # Lakukan clustering dengan jumlah cluster optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Tambahkan hasil cluster ke dataframe asli
    data['Cluster'] = clusters

    # Visualisasi hasil dengan PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    plt.figure(figsize=(12, 8))
    for cluster in range(optimal_k):
        plt.scatter(
            data_pca[clusters == cluster, 0],
            data_pca[clusters == cluster, 1],
            label=f'Cluster {cluster}'
        )

    plt.title('Segmentasi Pasien dengan KMeans dan PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Analisis karakteristik tiap cluster
    cluster_analysis = data.groupby('Cluster')[features].mean()

    # Heatmap untuk perbandingan cluster
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_analysis, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Karakteristik Rata-rata per Cluster')
    plt.show()

    return data, cluster_analysis

def prediksi_risiko_readmisi(data, target='Readmisi_30Hari'):
    """
    Membangun model prediksi risiko readmisi pasien dalam 30 hari
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

    # Pilih fitur dan target
    features = ['Umur', 'Jenis_Kelamin', 'Lama_Rawat', 'Diagnosis_Utama',
                'Jumlah_Obat', 'Riwayat_Readmisi', 'Tekanan_Darah_Sistolik',
                'Glukosa_Darah', 'BMI']

    # One-hot encoding untuk fitur kategorikal
    data_model = pd.get_dummies(data[features + [target]], columns=['Jenis_Kelamin', 'Diagnosis_Utama'])

    # Split data
    X = data_model.drop(target, axis=1)
    y = data_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    return model, importances, X.columns
```

### Geospatial Analysis

Analisis geospasial menggunakan data yang memiliki komponen geografis untuk mengidentifikasi pola, tren, dan hubungan dalam konteks spasial.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def visualisasi_peta_heatmap(data, lat_col='latitude', lon_col='longitude',
                             value_col=None, radius=15):
    """
    Membuat heatmap dari data geospasial
    """
    # Buat peta dasar
    center = [data[lat_col].mean(), data[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=12)

    # Siapkan data untuk heatmap
    if value_col:
        heat_data = [[row[lat_col], row[lon_col], row[value_col]]
                      for _, row in data.iterrows()]
    else:
        heat_data = [[row[lat_col], row[lon_col]]
                      for _, row in data.iterrows()]

    # Tambahkan heatmap ke peta
    HeatMap(heat_data, radius=radius).add_to(m)

    return m

def clustering_spasial(data, lat_col='latitude', lon_col='longitude',
                        eps=0.5, min_samples=5):
    """
    Melakukan clustering spasial dengan DBSCAN
    """
    # Ekstrak koordinat
    coords = data[[lat_col, lon_col]].values

    # Normalisasi data
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Lakukan clustering dengan DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords_scaled)

    # Tambahkan hasil cluster ke dataframe
    data['cluster'] = clusters

    # Visualisasi hasil clustering
    m = folium.Map(location=[data[lat_col].mean(), data[lon_col].mean()],
                   zoom_start=12)

    # Warna untuk cluster
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']

    # Tambahkan marker untuk setiap titik
    for cluster_id in set(clusters):
        if cluster_id == -1:
            # Noise points
            cluster_data = data[data['cluster'] == cluster_id]
            cluster_name = "Noise"
            color = 'gray'
        else:
            cluster_data = data[data['cluster'] == cluster_id]
            cluster_name = f"Cluster {cluster_id}"
            color = colors[cluster_id % len(colors)]

        # Buat marker cluster untuk titik-titik dalam cluster yang sama
        marker_cluster = MarkerCluster(name=cluster_name).add_to(m)

        for _, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Cluster: {cluster_name}"
            ).add_to(marker_cluster)

    # Tambahkan layer control
    folium.LayerControl().add_to(m)

    return m, data

def analisis_jarak_terdekat(data, poin_referensi, lat_col='latitude', lon_col='longitude', k=5):
    """
    Menemukan k titik terdekat dari titik referensi
    """
    from sklearn.neighbors import NearestNeighbors

    # Ekstrak koordinat
    coords = data[[lat_col, lon_col]].values

    # Inisialisasi model nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)

    # Temukan k titik terdekat
    distances, indices = nbrs.kneighbors([poin_referensi])

    # Ambil data titik terdekat
    nearest_points = data.iloc[indices[0][1:]]  # Indeks 0 adalah titik referensi itu sendiri

    # Visualisasi hasil
    m = folium.Map(location=poin_referensi, zoom_start=14)

    # Tambahkan marker untuk titik referensi
    folium.Marker(
        location=poin_referensi,
        popup="Titik Referensi",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Tambahkan marker untuk titik-titik terdekat
    for i, (_, row) in enumerate(nearest_points.iterrows()):
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=f"#{i+1} Terdekat - Jarak: {distances[0][i+1]:.2f} unit",
            icon=folium.Icon(color='blue')
        ).add_to(m)

        # Tambahkan garis antara titik referensi dan titik terdekat
        folium.PolyLine(
            locations=[poin_referensi, [row[lat_col], row[lon_col]]],
            color='green',
            weight=2,
            opacity=0.7
        ).add_to(m)

    return m, nearest_points, distances[0][1:]
```

### Customer Analytics

Customer analytics menggunakan data pelanggan untuk mendapatkan wawasan tentang perilaku pelanggan, preferensi, dan pola pembelian untuk meningkatkan strategi bisnis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

def analisis_rfm(data, id_col='customer_id', date_col='date', monetary_col='amount'):
    """
    Melakukan analisis RFM (Recency, Frequency, Monetary) untuk segmentasi pelanggan
    """
    # Pastikan kolom tanggal dalam format datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    # Tanggal analisis (tanggal terakhir dalam data + 1 hari)
    current_date = data[date_col].max() + pd.Timedelta(days=1)

    # Hitung metrik RFM
    rfm_data = data.groupby(id_col).agg({
        date_col: lambda x: (current_date - x.max()).days,  # Recency
        id_col: 'count',  # Frequency
        monetary_col: 'sum'  # Monetary
    }).reset_index()

    # Rename kolom
    rfm_data.rename(columns={
        date_col: 'recency',
        id_col: 'frequency',
        monetary_col: 'monetary'
    }, inplace=True)

    # Tambahkan monetary_avg
    rfm_data['monetary_avg'] = rfm_data['monetary'] / rfm_data['frequency']

    # Hitung skor RFM (1-5, 5 adalah terbaik)
    rfm_data['R_score'] = pd.qcut(rfm_data['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_data['F_score'] = pd.qcut(rfm_data['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_data['M_score'] = pd.qcut(rfm_data['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    # Hitung skor RFM gabungan
    rfm_data['RFM_score'] = rfm_data['R_score'].astype(int) + rfm_data['F_score'].astype(int) + rfm_data['M_score'].astype(int)

    # Segmentasi berdasarkan skor RFM
    def segmentasi_pelanggan(row):
        if row['RFM_score'] >= 13:
            return 'Champions'
        elif 10 <= row['RFM_score'] < 13:
            return 'Loyal Customers'
        elif 7 <= row['RFM_score'] < 10:
            return 'Potential Loyalists'
        elif 5 <= row['RFM_score'] < 7:
            return 'At Risk Customers'
        else:
            return 'Lost Customers'

    rfm_data['segment'] = rfm_data.apply(segmentasi_pelanggan, axis=1)

    # Visualisasi distribusi segmen
    plt.figure(figsize=(10, 6))
    segment_counts = rfm_data['segment'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values)
    plt.title('Distribusi Segmen Pelanggan')
    plt.xlabel('Segmen')
    plt.ylabel('Jumlah Pelanggan')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualisasi karakteristik tiap segmen
    plt.figure(figsize=(12, 8))
    segment_stats = rfm_data.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    })

    # Normalisasi untuk visualisasi
    segment_stats_norm = segment_stats.copy()
    for col in segment_stats_norm.columns:
        segment_stats_norm[col] = (segment_stats_norm[col] - segment_stats_norm[col].min()) / (segment_stats_norm[col].max() - segment_stats_norm[col].min())

    segment_stats_norm = segment_stats_norm.reset_index().melt(id_vars='segment', var_name='metric', value_name='value')

    sns.barplot(x='segment', y='value', hue='metric', data=segment_stats_norm)
    plt.title('Karakteristik Segmen Pelanggan (Normalized)')
    plt.xlabel('Segmen')
    plt.ylabel('Nilai (Normalized)')
    plt.xticks(rotation=45)
    plt.legend(title='Metrik')
    plt.tight_layout()
    plt.show()

    return rfm_data, segment_stats

def prediksi_ltv(data, id_col='customer_id', date_col='date', monetary_col='amount', time_period=12):
    """
    Memprediksi Customer Lifetime Value menggunakan model BG/NBD dan Gamma-Gamma
    """
    # Pastikan kolom tanggal dalam format datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    # Siapkan data summary
    summary = summary_data_from_transaction_data(
        data,
        customer_id_col=id_col,
        datetime_col=date_col,
        monetary_value_col=monetary_col
    )

    # Fit BG/NBD model untuk memprediksi frekuensi pembelian masa depan
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        summary['frequency'],
        summary['recency'],
        summary['T']
    )

    # Prediksi berapa kali pembelian di masa depan
    summary['predicted_purchases'] = bgf.predict(time_period,
                                               summary['frequency'],
                                               summary['recency'],
                                               summary['T'])

    # Fit Gamma-Gamma model untuk memprediksi nilai pembelian rata-rata
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        summary['frequency'],
        summary['monetary_value'],
        frequency_greater_than=0
    )

    # Prediksi nilai pembelian rata-rata
    summary['predicted_avg_purchase'] = ggf.conditional_expected_average_profit(
        summary['frequency'],
        summary['monetary_value']
    )

    # Hitung Customer Lifetime Value (CLV)
    summary['clv'] = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=time_period,
        discount_rate=0.01
    )

    # Visualisasi distribusi CLV
    plt.figure(figsize=(10, 6))
    sns.histplot(summary['clv'], kde=True)
    plt.title(f'Distribusi Customer Lifetime Value ({time_period} bulan)')
    plt.xlabel('CLV')
    plt.ylabel('Jumlah Pelanggan')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Identifikasi pelanggan dengan nilai tertinggi
    top_customers = summary.sort_values('clv', ascending=False).head(10)

    return summary, bgf, ggf, top_customers

def analisis_churn(data, id_col='customer_id', date_col='date', churn_threshold_days=90):
    """
    Menganalisis churn pelanggan dan membangun model prediksi
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve

    # Pastikan kolom tanggal dalam format datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    # Tanggal analisis (tanggal terakhir dalam data)
    current_date = data[date_col].max()

    # Hitung tanggal transaksi terakhir untuk setiap pelanggan
    last_transaction = data.groupby(id_col)[date_col].max().reset_index()
    last_transaction['days_since_last_purchase'] = (current_date - last_transaction[date_col]).dt.days

    # Tandai pelanggan yang churn (tidak bertransaksi dalam threshold hari terakhir)
    last_transaction['is_churned'] = last_transaction['days_since_last_purchase'] > churn_threshold_days

    # Gabungkan dengan fitur pelanggan (contoh fitur sederhana dari data transaksi)
    customer_features = data.groupby(id_col).agg({
        date_col: ['count', lambda x: (x.max() - x.min()).days],
        monetary_col: ['sum', 'mean', 'std']
    }).reset_index()

    # Flatten multi-level columns
    customer_features.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in customer_features.columns
    ]

    # Rename kolom untuk kejelasan
    customer_features.rename(columns={
        f"{date_col}_count": "frequency",
        f"{date_col}_<lambda>": "tenure_days",
        f"{monetary_col}_sum": "total_spent",
        f"{monetary_col}_mean": "avg_order_value",
        f"{monetary_col}_std": "order_value_std"
    }, inplace=True)

    # Tambahkan fitur turunan
    customer_features['purchase_frequency'] = customer_features['frequency'] / (customer_features['tenure_days'] + 1)  # +1 untuk menghindari div by zero

    # Gabungkan dengan label churn
    customer_data = pd.merge(customer_features, last_transaction[[id_col, 'is_churned']], on=id_col)

    # Visualisasi churn rate
    plt.figure(figsize=(8, 6))
    churn_rate = customer_data['is_churned'].mean() * 100
    plt.pie([churn_rate, 100-churn_rate], labels=['Churned', 'Active'], autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightgreen'])
    plt.title(f'Churn Rate (Threshold: {churn_threshold_days} hari)')
    plt.axis('equal')
    plt.show()

    # Siapkan fitur dan target untuk model prediksi
    features = [col for col in customer_data.columns if col not in [id_col, 'is_churned']]
    X = customer_data[features]
    y = customer_data['is_churned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve untuk Prediksi Churn')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Feature importance
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=features_df)
    plt.title('Feature Importance untuk Prediksi Churn')
    plt.tight_layout()
    plt.show()

    return model, customer_data, features_df
```

### Supply Chain Optimization

Supply Chain Optimization menggunakan data dan algoritma untuk meningkatkan efisiensi dalam rantai pasok, seperti manajemen persediaan, transportasi, dan peramalan permintaan.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def optimasi_persediaan(demand_history, holding_cost=0.2, stockout_cost=2.0,
                        order_cost=100, lead_time=2, forecast_periods=12):
    """
    Menerapkan model Economic Order Quantity (EOQ) dan Safety Stock untuk optimasi persediaan
    """
    # Hitung peramalan permintaan menggunakan Holt-Winters
    model = ExponentialSmoothing(
        demand_history,
        trend='add',
        seasonal='add',
        seasonal_periods=min(len(demand_history)//2, 12)  # Asumsi seasonal_periods=12 untuk data bulanan
    )
    model_fit = model.fit(optimized=True)
    demand_forecast = model_fit.forecast(forecast_periods)

    # Visualisasi permintaan historis dan peramalan
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(demand_history)), demand_history, label='Permintaan Historis')
    plt.plot(range(len(demand_history), len(demand_history) + forecast_periods),
             demand_forecast, label='Peramalan', linestyle='--')
    plt.axhline(y=demand_history.mean(), color='r', linestyle='-', alpha=0.3, label='Rata-rata Historis')
    plt.title('Permintaan Historis dan Peramalan')
    plt.xlabel('Periode')
    plt.ylabel('Jumlah')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Hitung EOQ (Economic Order Quantity)
    annual_demand = demand_forecast.mean() * 12  # Konversi ke permintaan tahunan
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)

    # Hitung Safety Stock berdasarkan standar deviasi permintaan selama lead time
    demand_std = demand_history.std()
    service_level = 0.95  # 95% service level
    z_score = 1.645  # z-score untuk 95% service level
    safety_stock = z_score * demand_std * np.sqrt(lead_time)

    # Hitung Reorder Point (ROP)
    avg_daily_demand = demand_forecast.mean()
    reorder_point = (avg_daily_demand * lead_time) + safety_stock

    # Hitung Total Cost
    annual_ordering_cost = (annual_demand / eoq) * order_cost
    annual_holding_cost = (eoq / 2 + safety_stock) * holding_cost
    total_annual_cost = annual_ordering_cost + annual_holding_cost

    # Visualisasi hubungan antara ukuran pesanan dan biaya
    order_quantities = np.linspace(eoq * 0.5, eoq * 1.5, 100)
    ordering_costs = (annual_demand / order_quantities) * order_cost
    holding_costs = (order_quantities / 2 + safety_stock) * holding_cost
    total_costs = ordering_costs + holding_costs

    plt.figure(figsize=(10, 6))
    plt.plot(order_quantities, ordering_costs, label='Biaya Pemesanan')
    plt.plot(order_quantities, holding_costs, label='Biaya Penyimpanan')
    plt.plot(order_quantities, total_costs, label='Total Biaya')
    plt.axvline(x=eoq, color='r', linestyle='--', label=f'EOQ = {eoq:.2f}')
    plt.title('Trade-off Biaya Persediaan')
    plt.xlabel('Ukuran Pesanan')
    plt.ylabel('Biaya')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Hasil optimasi
    results = {
        'EOQ': eoq,
        'Safety_Stock': safety_stock,
        'Reorder_Point': reorder_point,
        'Total_Annual_Cost': total_annual_cost,
        'Demand_Forecast': demand_forecast
    }

    return results

def optimasi_rute_distribusi(lokasi_nodes, jarak_matrix, kendaraan=3, kapasitas=1000):
    """
    Menyelesaikan Vehicle Routing Problem (VRP) untuk optimasi rute distribusi
    """
    import folium

    # Buat model optimasi menggunakan PuLP
    model = LpProblem(name="Vehicle_Routing_Problem", sense=LpMinimize)

    # Jumlah node (termasuk depot)
    n = len(lokasi_nodes)
    depot = 0  # Asumsikan node 0 adalah depot

    # Buat variabel keputusan biner untuk setiap rute dan kendaraan
    # x[i,j,k] = 1 jika kendaraan k melakukan perjalanan dari i ke j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:  # Tidak ada perjalanan dari node ke dirinya sendiri
                for k in range(kendaraan):
                    x[i,j,k] = LpVariable(f'x_{i}_{j}_{k}', cat='Binary')

    # Variabel untuk mencatat urutan kunjungan
    u = {}
    for i in range(1, n):  # Hanya untuk node pelanggan (bukan depot)
        u[i] = LpVariable(f'u_{i}', lowBound=1, upBound=n-1, cat='Integer')

    # Fungsi tujuan: Minimasi total jarak tempuh
    model += lpSum(jarak_matrix[i][j] * x[i,j,k] for i in range(n) for j in range(n) if i != j for k in range(kendaraan))

    # Constraint 1: Setiap pelanggan harus dikunjungi tepat sekali
    for j in range(1, n):  # Untuk setiap pelanggan (bukan depot)
        model += lpSum(x[i,j,k] for i in range(n) if i != j for k in range(kendaraan)) == 1

    # Constraint 2: Setiap kendaraan harus berangkat dari depot
    for k in range(kendaraan):
        model += lpSum(x[depot,j,k] for j in range(1, n)) <= 1  # Maksimal 1 keberangkatan per kendaraan

    # Constraint 3: Flow conservation - Jika masuk ke node j, harus keluar dari node j
    for k in range(kendaraan):
        for j in range(n):
            model += lpSum(x[i,j,k] for i in range(n) if i != j) == lpSum(x[j,i,k] for i in range(n) if i != j)

    # Constraint 4: Subtour elimination menggunakan MTZ formulation
    M = n  # Big M value
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                for k in range(kendaraan):
                    model += u[i] - u[j] + M * x[i,j,k] <= M - 1

    # Selesaikan model
    model.solve()

    print(f"Status: {LpStatus[model.status]}")

    # Ekstrak rute optimal
    rute_optimal = {k: [] for k in range(kendaraan)}

    if LpStatus[model.status] == "Optimal":
        for k in range(kendaraan):
            node_sekarang = depot
            rute = [depot]

            while True:
                ditemukan = False
                for j in range(n):
                    if node_sekarang != j and x[node_sekarang,j,k].value() > 0.5:
                        rute.append(j)
                        node_sekarang = j
                        ditemukan = True
                        break

                if not ditemukan or node_sekarang == depot:
                    break

            # Jika rute tidak kembali ke depot, tambahkan depot di akhir
            if rute[-1] != depot:
                rute.append(depot)

            # Simpan rute jika memiliki setidaknya 3 node (depot -> pelanggan -> depot)
            if len(rute) >= 3:
                rute_optimal[k] = rute

    # Visualisasi rute optimal
    # Buat peta dengan folium
    map_center = [np.mean([lokasi[0] for lokasi in lokasi_nodes]),
                  np.mean([lokasi[1] for lokasi in lokasi_nodes])]
    m = folium.Map(location=map_center, zoom_start=12)

    # Warna untuk rute berbeda
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']

    # Tambahkan marker untuk semua lokasi
    for i, lokasi in enumerate(lokasi_nodes):
        if i == depot:
            folium.Marker(
                location=lokasi,
                popup=f"Depot",
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
        else:
            folium.Marker(
                location=lokasi,
                popup=f"Pelanggan {i}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

    # Tambahkan rute ke peta
    for k, rute in rute_optimal.items():
        if rute:
            lintasan = []
            for node in rute:
                lintasan.append(lokasi_nodes[node])

            folium.PolyLine(
                locations=lintasan,
                color=colors[k % len(colors)],
                weight=2.5,
                opacity=0.8,
                popup=f"Rute Kendaraan {k+1}"
            ).add_to(m)

    # Hitung total jarak
    total_jarak = 0
    for k, rute in rute_optimal.items():
        if len(rute) >= 2:
            for i in range(len(rute)-1):
                total_jarak += jarak_matrix[rute[i]][rute[i+1]]

    return m, rute_optimal, total_jarak

def peramalan_permintaan_multi_level(data, produk_hierarchy, forecast_periods=12):
    """
    Melakukan peramalan permintaan multi-level untuk supply chain
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Pastikan data dalam format yang benar
    data_copy = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data_copy.index):
        data_copy.index = pd.DatetimeIndex(data_copy.index)

    # Dictionary untuk menyimpan hasil peramalan
    forecast_results = {}

    # Peramalan untuk setiap level dalam hierarchy
    for level, products in produk_hierarchy.items():
        print(f"Meramalkan untuk level: {level}")

        for product in products:
            print(f"  Produk: {product}")

            # Ambil data historis produk
            if product in data_copy.columns:
                history = data_copy[product]

                # Split data untuk training dan testing
                train_size = int(len(history) * 0.8)
                train, test = history[:train_size], history[train_size:]

                # Coba beberapa model SARIMA dan pilih yang terbaik
                best_mae = float('inf')
                best_model = None
                best_order = None
                best_seasonal_order = None

                # Grid search sederhana untuk parameter
                for p in [0, 1]:
                    for d in [0, 1]:
                        for q in [0, 1]:
                            for P in [0, 1]:
                                for D in [0, 1]:
                                    for Q in [0, 1]:
                                        try:
                                            # Gunakan periode musiman 12 untuk data bulanan
                                            # Sesuaikan dengan frekuensi data yang sebenarnya
                                            model = SARIMAX(
                                                train,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, 12),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                            )
                                            model_fit = model.fit(disp=False)

                                            # Forecast untuk validation set
                                            forecast = model_fit.forecast(steps=len(test))

                                            # Evaluasi
                                            mae = mean_absolute_error(test, forecast)

                                            if mae < best_mae:
                                                best_mae = mae
                                                best_model = model_fit
                                                best_order = (p, d, q)
                                                best_seasonal_order = (P, D, Q, 12)
                                        except:
                                            continue

                if best_model is not None:
                    print(f"    Best SARIMA model: {best_order}, {best_seasonal_order}")

                    # Fit model dengan seluruh data
                    final_model = SARIMAX(
                        history,
                        order=best_order,
                        seasonal_order=best_seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    final_model_fit = final_model.fit(disp=False)

                    # Forecast ke depan
                    forecast = final_model_fit.forecast(steps=forecast_periods)
                    forecast_results[product] = {
                        'history': history,
                        'forecast': forecast,
                        'model': final_model_fit,
                        'mae': best_mae
                    }

                    # Plot forecast
                    plt.figure(figsize=(12, 6))
                    plt.plot(history.index, history, label='Historical')

                    # Buat indeks untuk forecast
                    forecast_index = pd.date_range(
                        start=history.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq=history.index.freq or pd.infer_freq(history.index)
                    )

                    plt.plot(forecast_index, forecast, label='Forecast', color='red')
                    plt.fill_between(
                        forecast_index,
                        forecast - 1.96 * np.sqrt(final_model_fit.params['sigma2']),
                        forecast + 1.96 * np.sqrt(final_model_fit.params['sigma2']),
                        color='pink', alpha=0.3
                    )

                    plt.title(f'Forecast for {product}')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

    return forecast_results
```

# Topik Lanjutan Data Science

## 23. Data Privacy dan Responsible AI

Data Privacy (Privasi Data) dan Responsible AI (AI yang Bertanggung Jawab) menjadi aspek penting dalam pengembangan solusi data science modern, terutama di era dengan peraturan privasi yang semakin ketat.

### Differential Privacy

Differential Privacy adalah teknik matematika yang memungkinkan analisis data sambil menjaga privasi individu. Teknik ini bekerja dengan menambahkan "noise" (kebisingan) secara sistematis ke dalam data atau hasil analisis.

```python
# Contoh implementasi Differential Privacy sederhana menggunakan library PyDP
import pydp as dp
from pydp.algorithms.laplacian import BoundedSum

# Data asli
data = [1, 2, 3, 4, 5]

# Membuat objek BoundedSum dengan epsilon (parameter privasi)
epsilon = 1.0
lower_bound = 1
upper_bound = 5
sum_algorithm = BoundedSum(epsilon=epsilon, lower_bound=lower_bound, upper_bound=upper_bound)

# Menghitung jumlah dengan differential privacy
private_sum = sum_algorithm.quick_result(data)
print(f"Jumlah asli: {sum(data)}")
print(f"Jumlah dengan differential privacy: {private_sum}")
```

### Federated Learning

Federated Learning adalah pendekatan pembelajaran mesin di mana model dilatih pada perangkat lokal yang berisi data pengguna, tanpa perlu mengirimkan data tersebut ke server pusat. Hanya parameter model yang dikirim, bukan data mentah.

```python
# Contoh konseptual Federated Learning menggunakan TensorFlow Federated
import tensorflow as tf
import tensorflow_federated as tff

# Mendefinisikan model sederhana
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Setup untuk TFF
def model_fn():
    model = create_model()
    return tff.learning.from_keras_model(
        model,
        input_spec=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Inisialisasi algoritma federated averaging
fed_avg = tff.learning.build_federated_averaging_process(model_fn)
```

### Model Auditing

Model Auditing melibatkan evaluasi sistematis terhadap model machine learning untuk mengidentifikasi potensi bias, diskriminasi, atau masalah etika lainnya.

```python
# Contoh audit model untuk bias menggunakan AI Fairness 360
import aif360.datasets as datasets
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Load dataset
dataset = datasets.AdultDataset()
privileged_groups = [{'sex': 1}]  # laki-laki
unprivileged_groups = [{'sex': 0}]  # perempuan

# Hitung metrik bias
metric = BinaryLabelDatasetMetric(dataset,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)

# Statistical Parity Difference (perbedaan proporsi hasil positif)
print(f"Disparate Impact: {metric.disparate_impact()}")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")
```

### Transparency Tools

Tools transparansi membantu menjelaskan bagaimana model AI membuat keputusan, meningkatkan kepercayaan dan pemahaman.

```python
# Contoh penggunaan SHAP untuk transparansi model
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Dataset dan model sederhana
X = pd.DataFrame(np.random.random((100, 5)), columns=[f'fitur_{i}' for i in range(5)])
y = (X['fitur_0'] > 0.5).astype(int)

model = RandomForestClassifier().fit(X, y)

# Menjelaskan prediksi model menggunakan SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualisasi penjelasan
shap.summary_plot(shap_values, X)
```

### Regulatory Compliance

Kepatuhan terhadap regulasi seperti GDPR (Eropa), CCPA (California), atau peraturan data lainnya sangat penting untuk aplikasi data science yang legal dan etis.

#### Teknik Utama untuk Regulatory Compliance:

1. **Data Minimization**: Hanya mengumpulkan data yang benar-benar diperlukan
2. **Right to Erasure**: Memastikan data dapat dihapus jika diminta
3. **Data Portability**: Memungkinkan pengguna mengakses dan memindahkan data mereka
4. **Consent Management**: Mengelola persetujuan pengguna untuk penggunaan data

```python
# Contoh pseudocode untuk manajemen persetujuan
class ConsentManager:
    def __init__(self):
        self.user_consents = {}

    def record_consent(self, user_id, purpose, has_consented):
        if user_id not in self.user_consents:
            self.user_consents[user_id] = {}
        self.user_consents[user_id][purpose] = {
            'consented': has_consented,
            'timestamp': datetime.now()
        }

    def check_consent(self, user_id, purpose):
        if user_id not in self.user_consents:
            return False
        if purpose not in self.user_consents[user_id]:
            return False
        return self.user_consents[user_id][purpose]['consented']

    def delete_user_data(self, user_id):
        if user_id in self.user_consents:
            del self.user_consents[user_id]
            # Delete associated data as well
            return True
        return False
```

## 24. Data Science Infrastructure

Infrastruktur Data Science yang baik sangat penting untuk mendukung siklus hidup pengembangan model ML/AI dari awal hingga produksi.

### Feature Stores

Feature Store adalah komponen infrastruktur yang menyimpan, mengelola, dan melayani fitur untuk training dan inferensi. Manfaatnya termasuk konsistensi fitur dan penggunaan kembali kode.

```python
# Contoh penggunaan Feature Store dengan Feast
import feast
import pandas as pd
from datetime import datetime, timedelta

# Definisi Feature Store
customer_features = feast.FeatureView(
    name="customer_features",
    entities=["customer_id"],
    ttl=timedelta(days=30),
    features=[
        feast.Feature(name="total_orders", dtype=feast.ValueType.INT64),
        feast.Feature(name="avg_order_value", dtype=feast.ValueType.FLOAT)
    ],
    batch_source=feast.FileSource(path="customer_stats.parquet")
)

# Registrasi feature view
fs = feast.FeatureStore(repo_path=".")
fs.apply([customer_features])

# Ambil fitur untuk training
training_df = fs.get_historical_features(
    entity_df=pd.DataFrame({"customer_id": [1, 2, 3]}),
    features=["customer_features:total_orders", "customer_features:avg_order_value"]
).to_df()
```

### Model Registry

Model Registry adalah repositori terpusat yang melacak versi model, metadata, dan artefak. Ini membantu dalam tata kelola model dan deployment.

```python
# Contoh penggunaan MLflow Model Registry
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Membuat dan melatih model
X = np.random.random((100, 4))
y = X[:, 0] + 2 * X[:, 1]
model = RandomForestRegressor()
model.fit(X, y)

# Log model ke MLflow
with mlflow.start_run():
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("training_rmse", 0.05)
    mlflow.sklearn.log_model(model, "model")

    # Daftarkan model ke registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(model_uri, "RandomForestRegressor")
    print(f"Model didaftarkan sebagai: {mv.name} versi {mv.version}")
```

### Experiment Tracking

Experiment Tracking memungkinkan ilmuwan data melacak, membandingkan, dan memvisualisasikan eksperimen machine learning.

```python
# Contoh experiment tracking dengan MLflow
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Konfigurasi eksperimen
mlflow.set_experiment("Eksperimen-NN")

# Parameter model
learning_rate = 0.01
epochs = 10
hidden_size = 64

# Mencatat eksperimen
with mlflow.start_run():
    # Log parameter
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("hidden_size", hidden_size)

    # Membuat model sederhana
    model = nn.Sequential(
        nn.Linear(10, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )

    # Simulasi training
    for epoch in range(epochs):
        loss_value = 1.0 / (epoch + 1)  # Simulasi loss yang menurun
        mlflow.log_metric("loss", loss_value, step=epoch)

    # Simpan model
    mlflow.pytorch.log_model(model, "model")
```

### Resource Management

Resource Management memastikan bahwa komputasi data science (CPU, GPU, memori) dialokasikan secara efisien.

```python
# Contoh penggunaan Ray untuk resource management
import ray

# Inisialisasi Ray
ray.init()

# Mendefinisikan fungsi yang dapat dijalankan secara paralel
@ray.remote(num_cpus=2, num_gpus=0.5)  # Alokasi resource
def process_batch(batch_data):
    # Proses data intensif
    import time
    time.sleep(1)  # Simulasi pekerjaan
    return len(batch_data)

# Jalankan secara paralel dengan manajemen resource otomatis
batch_size = 100
batches = [list(range(i, i+batch_size)) for i in range(0, 1000, batch_size)]
futures = [process_batch.remote(batch) for batch in batches]

# Tunggu hasil
results = ray.get(futures)
print(f"Jumlah batch yang diproses: {len(results)}")
```

### Scalable Pipelines

Pipeline yang scalable memungkinkan pemrosesan data dan inferensi model pada skala besar.

```python
# Contoh pipeline scalable dengan Apache Beam
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Definisi tahapan pipeline
class ExtractFeatures(beam.DoFn):
    def process(self, element):
        user_id, data = element
        # Ekstrak fitur dari data mentah
        features = {
            'feature1': data['value1'] / 100.0,
            'feature2': len(data['text']),
            'feature3': data['count'] * 2
        }
        return [(user_id, features)]

class PredictWithModel(beam.DoFn):
    def setup(self):
        # Load model saat worker mulai
        import pickle
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def process(self, element):
        user_id, features = element
        # Buat prediksi
        feature_vector = [features['feature1'], features['feature2'], features['feature3']]
        prediction = self.model.predict([feature_vector])[0]
        return [(user_id, prediction)]

# Membuat dan menjalankan pipeline
options = PipelineOptions(['--runner=DirectRunner'])
with beam.Pipeline(options=options) as p:
    input_data = [
        ('user1', {'value1': 55, 'text': 'hello', 'count': 3}),
        ('user2', {'value1': 90, 'text': 'data science', 'count': 1})
    ]

    (p
     | 'CreateInputs' >> beam.Create(input_data)
     | 'ExtractFeatures' >> beam.ParDo(ExtractFeatures())
     | 'PredictWithModel' >> beam.ParDo(PredictWithModel())
     | 'PrintResults' >> beam.Map(print)
    )
```

## 25. Uncertainty Quantification

Uncertainty Quantification (Kuantifikasi Ketidakpastian) adalah bidang yang fokus pada pengukuran dan pengelolaan ketidakpastian dalam prediksi model machine learning.

### Bayesian Methods

Metode Bayesian menggunakan probabilitas untuk mewakili ketidakpastian dalam model dan prediksi.

```python
# Contoh regresi Bayesian dengan PyMC3
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate data sintetis
np.random.seed(42)
x = np.random.uniform(low=0, high=10, size=50)
y = 2 * x + np.random.normal(scale=2.0, size=50)

# Model Bayesian
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Fungsi linier
    mu = alpha + beta * x

    # Likelihood
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

    # Inferensi
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# Plot hasil
with model:
    pm.plot_trace(trace)

# Prediksi dengan ketidakpastian
x_new = np.linspace(0, 15, 100)
with model:
    pm.set_data({"x": x_new})
    posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y"], samples=1000)

# Plot prediksi dengan interval kepercayaan
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.6, label='Data')
plt.plot(x_new, 2 * x_new, color='green', label='True function')

# Plot prediksi dan interval kepercayaan 95%
pred_mean = posterior_pred['y'].mean(axis=0)
pred_std = posterior_pred['y'].std(axis=0)
plt.plot(x_new, pred_mean, color='red', label='Predicted')
plt.fill_between(x_new, pred_mean - 2*pred_std, pred_mean + 2*pred_std, color='red', alpha=0.2, label='95% CI')

plt.legend()
plt.title('Bayesian Linear Regression dengan Interval Ketidakpastian')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### Conformal Prediction

Conformal Prediction memberikan interval prediksi dengan jaminan statistik tentang cakupan.

```python
# Contoh Conformal Prediction dengan scikit-learn dan nonconformist
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from nonconformist.cp import IcpRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.nc import RegressorNc, AbsErrorErrFunc

# Data
X = np.random.rand(500, 10)
y = X[:, 0] * X[:, 1] + np.random.normal(0, 0.1, 500)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Conformal predictor
underlying_model = RegressorAdapter(model)
nc = RegressorNc(underlying_model, AbsErrorErrFunc())
icp = IcpRegressor(nc)

# Fit
underlying_model.fit(X_train, y_train)
icp.calibrate(X_cal, y_cal)

# Prediksi dengan interval
predictions = icp.predict(X_test, significance=0.05)

# Output interval prediksi
for i in range(5):
    lower, upper = predictions[i, 0], predictions[i, 1]
    true_value = y_test[i]
    print(f"Sampel {i}: Range prediksi [{lower:.3f}, {upper:.3f}], Nilai sebenarnya: {true_value:.3f}")
```

### Bootstrapping

Bootstrapping adalah teknik resampling yang digunakan untuk memperkirakan ketidakpastian statistik.

```python
# Contoh bootstrapping untuk estimasi interval kepercayaan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.normal(0, 2, 100)

# Model dasar
model = LinearRegression()
model.fit(X, y)
y_pred_base = model.predict(X)

# Bootstrap
n_bootstraps = 1000
bootstrap_coefs = []

for i in range(n_bootstraps):
    # Resample dengan penggantian
    indices = resample(range(len(X)), replace=True, n_samples=len(X))
    X_boot, y_boot = X[indices], y[indices]

    # Fit model
    model_boot = LinearRegression()
    model_boot.fit(X_boot, y_boot)

    # Simpan koefisien
    bootstrap_coefs.append(model_boot.coef_[0])

# Interval kepercayaan 95% untuk kemiringan
ci_lower = np.percentile(bootstrap_coefs, 2.5)
ci_upper = np.percentile(bootstrap_coefs, 97.5)

print(f"Koefisien dasar: {model.coef_[0]:.3f}")
print(f"Interval kepercayaan bootstrap 95%: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_base, color='red', label=f'Model dasar (slope={model.coef_[0]:.3f})')

# Plot beberapa model bootstrap
X_sorted = np.sort(X, axis=0)
for i in range(50):  # Plot 50 dari 1000 bootstrap models
    indices = resample(range(len(X)), replace=True, n_samples=len(X))
    X_boot, y_boot = X[indices], y[indices]
    model_boot = LinearRegression().fit(X_boot, y_boot)
    plt.plot(X_sorted, model_boot.predict(X_sorted), color='green', alpha=0.05)

plt.title('Regresi Linier dengan Bootstrap')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Monte Carlo Dropout

Monte Carlo Dropout menggunakan dropout selama inferensi untuk memperkirakan ketidakpastian model.

```python
# Contoh Monte Carlo Dropout dengan TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.2, size=X.shape)

# Membuat model dengan dropout
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.5),  # Dropout layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout layer
    tf.keras.layers.Dense(1)
])

# Compile dan latih
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

# Fungsi prediksi dengan MC Dropout
def mc_dropout_predict(model, X_test, n_iter=100):
    y_pred_list = []
    # Set dropout layer tetap aktif selama inferensi
    for i in range(n_iter):
        y_pred = model(X_test, training=True)  # training=True mempertahankan dropout
        y_pred_list.append(y_pred.numpy())

    # Gabungkan hasil dari semua iterasi
    y_pred_mean = np.mean(y_pred_list, axis=0)
    y_pred_std = np.std(y_pred_list, axis=0)

    return y_pred_mean, y_pred_std

# Prediksi dengan MC Dropout
X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
y_pred_mean, y_pred_std = mc_dropout_predict(model, X_test)

# Plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred_mean, 'r-', label='Prediksi')
plt.fill_between(X_test.flatten(),
                 y_pred_mean.flatten() - 2 * y_pred_std.flatten(),
                 y_pred_mean.flatten() + 2 * y_pred_std.flatten(),
                 alpha=0.3, color='red', label='Interval kepercayaan 95%')
plt.plot(X_test, np.sin(X_test), 'g--', label='Fungsi sebenarnya')
plt.title('Neural Network dengan Monte Carlo Dropout')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Ensemble Uncertainty

Ensemble Uncertainty menggunakan variasi prediksi di antara beberapa model untuk mengukur ketidakpastian.

```python
# Contoh Ensemble Uncertainty
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X.flatten()) + np.random.normal(0, 0.2, X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi individual dari setiap tree
predictions = np.array([tree.predict(X_test) for tree in model.estimators_])

# Hitung mean dan std dari prediksi ensemble
y_pred_mean = np.mean(predictions, axis=0)
y_pred_std = np.std(predictions, axis=0)

# Plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test data')
plt.plot(X_test, y_pred_mean, 'r-', label='Prediksi ensemble')
plt.fill_between(X_test.flatten(),
                 y_pred_mean - 2 * y_pred_std,
                 y_pred_mean + 2 * y_pred_std,
                 alpha=0.3, color='red', label='Interval ketidakpastian ensemble')
plt.plot(X_test, np.sin(X_test.flatten()), 'g--', label='Fungsi sebenarnya')
plt.title('Ketidakpastian Ensemble dari Random Forest')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Evaluasi ketidakpastian
# Menghitung apakah interval kepercayaan mencakup nilai sebenarnya
coverage = np.mean((y_test >= y_pred_mean - 2 * y_pred_std) &
                  (y_test <= y_pred_mean + 2 * y_pred_std))
print(f"Cakupan interval kepercayaan 95%: {coverage*100:.2f}%")

# Korelasi antara error absolut dan ketidakpastian yang diprediksi
abs_error = np.abs(y_test - y_pred_mean)
correlation = np.corrcoef(abs_error, y_pred_std)[0, 1]
print(f"Korelasi antara error dan ketidakpastian: {correlation:.3f}")
```
