def main():
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
if __name__ == "__main__":
    main()
