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
