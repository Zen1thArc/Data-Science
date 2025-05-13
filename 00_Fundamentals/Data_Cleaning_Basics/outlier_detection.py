def main():
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
if __name__ == "__main__":
    main()
