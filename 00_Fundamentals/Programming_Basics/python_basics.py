#!/usr/bin/env python3

def main():
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

if __name__ == "__main__":
    main()
