def main():

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

if __name__ == "__main__":
    main()
