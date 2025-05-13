#!/usr/bin/env python3

def main():
    # -- Mengambil semua kolom dari tabel
    SELECT * FROM employees;

    # -- Mengambil kolom tertentu
    SELECT employee_id, first_name, last_name FROM employees;

    # -- Menampilkan data unik
    SELECT DISTINCT department_id FROM employees;

    # -- Membatasi jumlah hasil
    SELECT * FROM employees LIMIT 10;

if __name__ == "__main__":
    main()
