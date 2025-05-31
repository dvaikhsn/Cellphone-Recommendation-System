# Laporan Proyek Machine Learning – Dava Ikhsan R

## 1. Project Overview


---

## 2. Business Understanding

### Problem Statements

1. Pengguna kesulitan memilih handphone dari ribuan pilihan yang tersedia di pasaran.
2. Sistem rekomendasi yang ada seringkali tidak memperhatikan preferensi personal atau spesifikasi yang diminati pengguna.

### Goals

1. Membangun sistem rekomendasi yang menyarankan handphone berdasarkan kesamaan spesifikasi (Content-Based).
2. Memberikan rekomendasi berdasarkan perilaku pengguna lain dengan preferensi serupa (Collaborative Filtering).

### Solution Approach

- **Content-Based Filtering**: Menggunakan data spesifikasi handphone seperti RAM, harga, OS, kamera, dll.
- **Collaborative Filtering (SVD)**: Memanfaatkan interaksi pengguna (rating) untuk menyarankan produk.

---

## 3. Data Understanding

Dataset terdiri dari tiga file:
- `cellphones_ratings.csv` – Data rating antara pengguna dan handphone
- `cellphones_users.csv` – Informasi demografi pengguna
- `cellphones_data.csv` – Spesifikasi lengkap produk handphone

### Fitur Dataset

- `user_id`: ID unik pengguna
- `cellphone_id`: ID unik produk
- `rating`: Rating dari pengguna (skala 1–5)
- `brand`: Merek handphone
- `model`: Model atau seri
- `price`: Harga handphone
- `RAM`, `camera`, `battery size`: Fitur teknis
- `operating system`: Sistem operasi

### EDA (Exploratory Data Analysis)

- Distribusi rating pengguna
- Produk dengan jumlah rating terbanyak
- Korelasi fitur numerik (RAM, harga, kamera)
- Distribusi harga handphone

---

## 4. Data Preparation

- Menghapus data duplikat
- Penanganan missing value:
  - `ratings_df`: Drop baris kosong
  - `users_df`, `products_df`: Isi dengan forward-fill
- Penggabungan dataset ke dalam satu `merged_df`
- **Feature Engineering**:
  - TF-IDF pada nama model
  - One-hot encoding untuk brand dan OS
  - Normalisasi fitur numerik

---

## 5. Modeling

### ✅ Content-Based Filtering

- Menggunakan cosine similarity antar fitur handphone
- Top-5 produk paling mirip ditampilkan
- Cocok untuk user baru (cold-start)

### ✅ Collaborative Filtering (SVD)

- Menggunakan library Surprise
- Rekomendasi berdasarkan user-user serupa
- Lebih personal

---

## 6. Evaluation

### Metrik Evaluasi

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

### Hasil Evaluasi

| Model                         | MAE   | RMSE  |
|------------------------------|-------|-------|
| Baseline (mean rating)       | 0.65  | 0.80  |
| Collaborative Filtering (SVD)| 0.42  | 0.53  |
| Neural Collaborative Filtering| ~0.40 | ~0.51 |

---

## 7. Kesimpulan dan Saran

### ✅ Kesimpulan



### ✅ Saran



---
