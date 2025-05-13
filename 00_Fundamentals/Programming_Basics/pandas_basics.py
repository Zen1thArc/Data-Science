#!/usr/bin/env python3

def main():
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


if __name__ == "__main__":
    main()
