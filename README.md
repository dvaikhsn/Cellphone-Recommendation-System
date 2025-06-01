
# Laporan Proyek Machine Learning – Dava Ikhsan R

---

##  Project Overview

Proyek ini bertujuan untuk membangun sistem rekomendasi smartphone menggunakan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering**. Sistem ini membantu pengguna menemukan produk smartphone yang relevan berdasarkan karakteristik produk dan perilaku pengguna.

Rekomendasi produk sangat penting dalam industri e-commerce karena meningkatkan pengalaman pengguna dan kemungkinan pembelian.

---

## Business Understanding

### Problem Statement
Pengguna kesulitan memilih smartphone yang sesuai di tengah banyaknya pilihan. Tanpa rekomendasi yang tepat, pengguna cenderung bingung atau malah tidak membeli produk.

### Goals
- Membangun sistem rekomendasi smartphone berbasis konten dan kolaborasi.
- Memberikan rekomendasi yang akurat untuk meningkatkan kepuasan pengguna.

### Solution Approach
1. **Content-Based Filtering**: Menggunakan atribut produk (brand, model, RAM, kamera, baterai, dan harga) untuk mengukur kesamaan antar produk.
2. **Collaborative Filtering (SVD)**: Menggunakan data rating dari pengguna terhadap produk untuk merekomendasikan smartphone yang disukai oleh pengguna dengan preferensi serupa.

---

## Data Understanding

### Sumber Data
- Dataset berisi informasi produk smartphone (brand, model, harga, RAM, kamera, baterai).
- Dataset rating berisi feedback pengguna terhadap produk.
- Data berasal dari sumber open source & simulasi pengguna.

### Fitur Data
- `brand`: Merek ponsel.
- `model`: Model ponsel.
- `RAM`: Ukuran RAM dalam GB.
- `battery size`: Kapasitas baterai.
- `main camera`: Megapiksel kamera utama.
- `price`: Harga smartphone.
- `user_id`, `cellphone_id`, `rating`: Data interaksi pengguna dan rating.

### Exploratory Data Analysis (EDA)
- Visualisasi distribusi harga dan RAM menunjukkan variasi produk.
- Korelasi fitur menunjukkan hubungan positif antara RAM dan harga.

---

## Data Preparation

- **Gabungkan** semua atribut penting menjadi satu representasi string untuk content-based.
- **Normalisasi** fitur numerik menggunakan MinMaxScaler.
- **Konversi dataset rating** ke format Surprise `Dataset.load_from_df`.
- Hilangkan data duplikat dan tangani missing value.

---

## Modeling

### Content-Based Filtering
- Gunakan **TF-IDF Vectorizer** untuk fitur gabungan produk.
- Hitung **cosine similarity** antar produk.
- Produk mirip ditentukan dari kemiripan vektor fitur.

```python
def recommend_by_product(product_name, top_n=5):
    ...
```

Contoh hasil:
```
Top 5 Produk Mirip 'Galaxy S22':
- Asus Zenfone 8 (Similarity: 0.99)
- Google Pixel 6 Pro (0.98)
- Xiaomi Poco F4 (0.98)
- Oppo Find X5 Pro (0.98)
- Xiaomi 12 Pro (0.98)
```

### Collaborative Filtering (SVD)
- Gunakan algoritma **Singular Value Decomposition (SVD)** dari library `surprise`.
- Train-test split 80:20 untuk evaluasi model.

```python
svd = SVD()
svd.fit(trainset)
predictions = svd.test(testset)
```

Rekomendasi untuk user_id = 0:
```
- iPhone XR (Prediksi Rating: 6.12)
- X80 Pro (Prediksi Rating: 5.74)
```

---

##  Evaluation

### Metrik Evaluasi:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error) *(opsional)*

```python
rmse_score = np.sqrt(mean_squared_error(y_true, y_est))
```

Hasil:
```
RMSE: 2.1936
```

📌 Interpretasi:
- RMSE dalam skala 1–10 menunjukkan performa cukup baik.
- Prediksi relatif dekat dengan rating aktual.

---

## Kesimpulan dan Saran

### Kesimpulan
- Sistem rekomendasi berhasil dibangun menggunakan dua pendekatan.
- Content-Based Filtering efektif dalam menemukan produk serupa.
- Collaborative Filtering mampu merekomendasikan produk yang relevan berdasarkan pola interaksi pengguna.

### 💡 Saran
- Tingkatkan model collaborative filtering dengan algoritma **KNN**, **Matrix Factorization**, atau **Deep Learning**.
- Gunakan data pengguna nyata untuk validasi lebih baik.
- Kombinasikan kedua pendekatan dalam model **Hybrid Recommendation System** untuk hasil lebih akurat.

---

## 📎 Lampiran & Referensi
- 📁 Dataset simulasi dan skrip dapat diakses melalui file `.ipynb` yang disertakan.
- 📚 Library: `pandas`, `sklearn`, `surprise`, `matplotlib`, `seaborn`
