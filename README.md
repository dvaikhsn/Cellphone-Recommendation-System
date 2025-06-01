# Laporan Proyek Machine Learning – Dava Ikhsan R

---

##  Project Overview

Proyek ini bertujuan untuk membangun sistem rekomendasi smartphone menggunakan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering**. Sistem ini membantu pengguna menemukan produk smartphone yang relevan berdasarkan karakteristik produk dan perilaku pengguna.

Rekomendasi produk sangat penting dalam industri e-commerce karena meningkatkan pengalaman pengguna dan kemungkinan pembelian.

---

## Business Understanding

### Problem Statement
- Bagaimana cara membantu pengguna menemukan smartphone yang sesuai dengan kebutuhan mereka di tengah banyaknya pilihan yang tersedia?
- Bagaimana sistem dapat memberikan rekomendasi yang relevan agar pengguna tidak kebingungan dalam mengambil keputusan pembelian?

### Goals
- Membangun sistem rekomendasi smartphone berbasis konten dan kolaborasi.
- Memberikan rekomendasi yang akurat untuk meningkatkan kepuasan pengguna.

### Solution Approach
1. **Content-Based Filtering**: Menggunakan atribut produk (brand, model, RAM, kamera, baterai, dan harga) untuk mengukur kesamaan antar produk.
2. **Collaborative Filtering (SVD)**: Menggunakan data rating dari pengguna terhadap produk untuk merekomendasikan smartphone yang disukai oleh pengguna dengan preferensi serupa.

---

## Data Understanding
Dataset yang digunakan adalah data Cellphones Recommendations dari Kaggle:

📊 **Sumber dataset**: [Cellphones Recommendations](https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations)

### 🗂️ Struktur Dataset

#### **Dataset Products_df**
Dari Dataset Products_df, terdapat 14 kolom dan 32 baris yang berisi informasi mengenai spesifikasi berbagai produk handphone:
- **`cellphone_id`**: ID unik untuk setiap produk handphone.  
- **`brand`**: Merek dari handphone (misalnya Samsung, Xiaomi, dll).  
- **`model`**: Nama atau tipe model dari handphone.  
- **`operating system`**: Sistem operasi yang digunakan (seperti Android, iOS, dll).  
- **`internal memory`**: Kapasitas memori internal (penyimpanan) dari handphone.  
- **`RAM`**: Kapasitas RAM handphone.  
- **`performance`**: Indikator performa secara umum, bisa berupa skor benchmark atau kategori performa.  
- **`main camera`**: Resolusi atau konfigurasi kamera utama.  
- **`selfie camera`**: Resolusi atau konfigurasi kamera depan (selfie).  
- **`battery size`**: Kapasitas baterai, biasanya dalam mAh.  
- **`screen size`**: Ukuran layar handphone dalam inci.  
- **`weight`**: Berat handphone dalam gram.  
- **`price`**: Harga dari produk handphone.  
- **`release date`**: Tanggal atau tahun rilis produk ke pasar.
  
    ![image](https://github.com/user-attachments/assets/3344a3a0-afd6-45ef-94e2-bfca3d2ee940)

Dari dataset `products_df` tidak terdapat nilai yang hilang dan duplikat, sehingga proses analisis dapat langsung dilakukan tanpa perlu menangani data kosong atau null.  

---
#### **Dataset ratings_df**
Dari dataset `ratings_df`, terdapat **3 kolom** dan **990 baris** yang berisi informasi mengenai:

1. **`user_id`** – ID unik dari masing-masing pengguna yang memberikan rating.
2. **`cellphone_id`** – ID unik dari masing-masing produk handphone yang diberi rating.
3. **`rating`** – Nilai penilaian yang diberikan oleh pengguna terhadap produk, dalam skala 1–10.
   
    ![image](https://github.com/user-attachments/assets/71d9b122-7def-42f9-a134-e1c2abc320b7)

Dari dataset `ratings_df` tidak terdapat nilai yang hilang dan duplikat, sehingga proses analisis dapat langsung dilakukan tanpa perlu menangani data kosong atau null. 

---
#### **Dataset users_df** 
Dari dataset `users_df`, terdapat **4 kolom** dan **99 baris** yang berisi informasi mengenai:

1. **`user_id`** – ID unik dari masing-masing pengguna.
2. **`age`** – Usia pengguna.
3. **`gender`** – Jenis kelamin pengguna (misalnya, 'M' untuk male dan 'F' untuk female).
4. **`occupation`** – Pekerjaan pengguna (dalam format kategori atau teks).

   ![image](https://github.com/user-attachments/assets/0663f357-443c-465d-8bdf-9212a68ad35a)

Dari dataset `users_df` hanya memiliki 1 nilai yang hilang pada kolom `occupation` dan tidak terdapat nilai yang duplikat.

---
## 📊 Exploratory Data Analysis (EDA)
### **Dataset Products_df**
![image](https://github.com/user-attachments/assets/61059972-b72f-4ac8-a2ac-407784e6ed15)

Dataset `products_df` berisi **33 baris** dan **14 kolom** yang menggambarkan spesifikasi berbagai model handphone. Berikut adalah beberapa insight awal berdasarkan analisis eksploratif:
- Semua kolom memiliki **nilai lengkap (non-null)** tanpa missing data.
- Terdapat 8 kolom bertipe numerik (int64 dan float64), seperti `internal memory`, `RAM`, `performance`, `main camera`, `selfie camera`, `battery size`, `screen size`, `weight`, dan `price`.
- Terdapat 4 kolom bertipe objek, yaitu `brand`, `model`, `operating system`, dan `release date`.
- Dataset relatif kecil (33 baris), sehingga analisis bisa lebih mendalam pada setiap fitur.

#### 1. Distribusi Harga Handphone
![image](https://github.com/user-attachments/assets/dfa906de-3c0e-4968-acdb-e45a13d1b1f8)
1. **Distribusi Positif Skew (Right-Skewed)**:
   - Sebagian besar handphone memiliki harga di bawah **1000**.
   - Jumlah produk menurun secara signifikan pada harga di atas **1000–1200**, menandakan hanya sedikit handphone yang berada di segmen high-end.

2. **Segmen Pasar Dominan**:
   - Banyak produk berada di rentang harga **100–500**, menandakan fokus pasar pada kategori **low-end hingga mid-range**.
   - Hal ini mengindikasikan bahwa produsen cenderung menargetkan pengguna dengan daya beli menengah ke bawah.

3. **Outlier Harga Tinggi**:
   - Terdapat beberapa handphone dengan harga ekstrem di atas **1750–2000**, yang tampak sebagai **produk premium** atau flagship.
   - Perlu dilakukan pertimbangan apakah harga ekstrem ini akan dipertahankan dalam modeling, atau dikelompokkan secara khusus.

4. **Kurva KDE**:
   - Kurva menunjukkan puncak (mode) pada rentang **200–400**, mengindikasikan harga paling umum dari produk yang tersedia.
  
#### 2. Korelasi antara fitur numerik
![image](https://github.com/user-attachments/assets/7937047d-60ee-4527-a069-bc9667e3bb54)

Heatmap menunjukkan nilai korelasi antara fitur-fitur numerik dalam dataset handphone, yaitu:
- `RAM`
- `main camera`
- `battery size`
- `price`

Nilai korelasi berkisar antara -1 hingga 1:
- Positif → fitur bergerak searah
- Negatif → fitur bergerak berlawanan arah
- 0 → tidak berkorelasi


1. **Harga Paling Berkorelasi dengan RAM**:
   - Korelasi tertinggi adalah antara `price` dan `RAM` (**0.71**), menunjukkan bahwa harga handphone cenderung meningkat seiring peningkatan RAM.

2. **Kamera dan Baterai**:
   - `main camera` memiliki korelasi sedang terhadap `battery size` (**0.60**), yang bisa jadi indikasi bahwa kamera yang lebih baik membutuhkan baterai yang lebih besar.

3. **Korelasi Negatif dengan Harga**:
   - `main camera` (-0.26) dan `battery size` (-0.13) justru menunjukkan **korelasi negatif terhadap harga**. Ini mengindikasikan bahwa:
     - Tidak semua handphone dengan kamera besar atau baterai besar memiliki harga tinggi.
     - Fitur seperti kamera dan baterai mungkin tidak menjadi indikator utama dalam penetapan harga, atau bisa jadi lebih dominan di segmen mid-end.

---
### **Dataset users_df** 
![image](https://github.com/user-attachments/assets/251e685b-f002-4052-a37b-b1bbbf712bdb)

Dataset `users_df` berisi 99 baris dan 4 kolom dengan Tipe data:
  - `user_id` (int64): ID unik tiap pengguna
  - `age` (int64): Usia pengguna
  - `gender` (object): Jenis kelamin pengguna
  - `occupation` (object): Pekerjaan pengguna
  - Kolom `occupation` memiliki **1 nilai yang hilang** (NaN).

Berdasarkan Deskripsi `users_df`,
  ```
  print("\nStatistik Deskriptif:")
  users_df.describe(include='all')
  ```
- **Usia pengguna:**
  - Rata-rata: 36.4 tahun
  - Minimum: 21 tahun
  - Maksimum: 61 tahun
  - Kuartil:
    - Q1: 29.5
    - Median: 33
    - Q3: 42

- **Pekerjaan:**
  - Total unik: 56 jenis
  - Paling umum: `Information Technology` (10 pengguna)
  - Terdapat **1 nilai yang hilang**

- **ID Pengguna (`user_id`):**
  - Nilai minimum: 0
  - Nilai maksimum: 258
  - Rata-rata: 136.37
 

#### 1. Visualisasi distribusi usia pengguna
![image](https://github.com/user-attachments/assets/576d4766-ccfe-49b4-ba50-406832f7bdc3)

Berdasarkan visualisasi dari distribusi pengguna, `users_df` memiliki informasi:
- Mayoritas pengguna berusia **25–35 tahun**.
- Distribusi condong ke kanan (right-skewed), artinya lebih banyak pengguna muda dibandingkan yang lebih tua.
- Sedikit pengguna yang berusia di atas 50 tahun.

#### 2. Visualisasi jumlah pengguna berdasarkan jenis kelamin
![image](https://github.com/user-attachments/assets/cb0bcb7a-6d06-4861-9de4-d23d9e2d304e)

Visualisasi tersebut, dapat di interpretasikan:
- Jumlah pengguna pria dan wanita hampir seimbang.
- **Pria sedikit lebih banyak** dibanding wanita (50 vs 46 pengguna).


#### 3. Visualisasi jumlah pengguna berdasarkan pekerjaan dengan horizontal bar chart
![image](https://github.com/user-attachments/assets/65b22215-bb7e-49bc-b554-a16b457fab26)

- Sebagian besar pengguna dalam dataset berasal dari latar belakang teknologi, dengan pekerjaan **Information Technology** menjadi yang paling umum, terlihat ada  **10 pengguna**. Hal ini menunjukkan potensi minat teknologi yang kuat di antara pengguna.
- Pekerjaan pengguna juga bervariasi, karena terdapat **56 kategori pekerjaan**
- Terdapat beberapa entri yang duplikat penulisan seperti `manager`, `Manager`, dan `Manager` → perlu **normalisasi teks**.

Maka dari itu karena terdapat entri yang duplikat, disini perlu di normalisasi dengan cara:
```
# Normalisasi kolom 'occupation' di users_df
users_df['occupation_cleaned'] = (
    users_df['occupation']
    .str.lower()              # ubah ke huruf kecil
    .str.strip()              # hilangkan spasi di awal/akhir
    .str.replace(r'[^a-z\s]', '', regex=True)  # hapus karakter non-huruf (opsional)
)

# Lihat hasil normalisasi dan distribusinya
occupation_counts = users_df['occupation_cleaned'].value_counts()
print(occupation_counts)

```
Sehingga didapatkan setelah normalisasi menjadi seperti ini:
![image](https://github.com/user-attachments/assets/8986bc48-1c7b-4f5b-8213-971adecc68b4)

---

### **Dataset ratings_df**
![image](https://github.com/user-attachments/assets/4ebc718d-5ccc-415f-9547-7acce9431b2d)

Dataset `users_df` berisi 990 baris dan 3 kolom dengan Tipe data:
  - `user_id` (int64): ID pengguna
  - `cellphone_id` (int64): ID handphone
  - `rating` (int64): Nilai rating dari pengguna

Dari dataset ini, semua kolom tidak ada yang memiliki nilai kosong.

Berdasarkan Deskripsi `ratings_df`,
```
ratings_df.describe(include='all')
```
![image](https://github.com/user-attachments/assets/04879538-3d08-47d1-801a-017451a89750)

Dapat di interpretasikan bahwa:
- **Rata-rata rating:** 6.7  
- **Median rating:** 7  
- **Rating minimum:** 1  
- **Rating maksimum:** 18  
- **Sebaran rating (IQR):** 5 hingga 9
- Kemudia terdapat outlier pada rating maksimum (18), dari skala 1–10. 

#### 1. Distribusi Nilai Rating
![image](https://github.com/user-attachments/assets/c326549c-8b6e-4ebd-8510-7c69e3631097)

Karena sebelumnya terdapat outlier, yaitu ada rating dengan nilai 18, maka dari itu:
```
# Buang data dengan rating di luar rentang 1–10
ratings_df = ratings_df[(ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 10)]
```
Sehingga dihasilkan seperti pada visualisasi di atas. Visual ini terlihat bahwa:
- Rating paling sering diberikan yaitu 8, terlhat hampir 200 kali.
- Sebagian besar pengguna memberikan **rating antara 6–10**, menunjukkan kecenderungan penilaian positif.

#### 2. Jumlah rating per produk
![image](https://github.com/user-attachments/assets/67982995-7439-4ce8-b81c-74a27f1066b7)

Jumlah rating per produk di atas terlihat bahwa:
- **iPhone 13 Pro Max** menjadi model yang paling banyak mendapatkan rating, yaitu lebih dari **40 rating**.
- Disusul oleh **Galaxy A32**, **Galaxy S22 Ultra**, dan **Galaxy Z Flip 3** yang masing-masing memiliki hampir **40 rating**.
- Secara keseluruhan, model dari **Apple**, **Samsung**, **Xiaomi**, dan **Motorola** mendominasi daftar 10 besar.
- Jumlah rating tinggi bisa mengindikasikan popularitas atau banyaknya pengguna yang terlibat dengan produk tersebut.

---

## Data Preparation
Langkah-langkah berikut dilakukan untuk menyiapkan data dari masing-masing dataset sebelum proses pemodelan. Semua teknik disusun sesuai urutan eksekusi di notebook:

### Data Preparation Products_df
Walaupun tidak ada nilai yang hilang dan duplikasi, dataset ini tetap dilakukan preparation sebagai berikut untuk memastikan data benar benar clear:
#### 1. Duplikasi Dihapus
- Seluruh baris duplikat dalam dataset `products_df` telah dihapus.
- Tindakan ini bertujuan untuk memastikan kembali  tidak ada entri ganda yang dapat mempengaruhi hasil analisis atau performa model.

#### 2. Penanganan Missing Value
- Baris dengan nilai kosong pada kolom `model` dan `price` dihapus karena kedua atribut ini bersifat krusial.
- Nilai kosong pada kolom `brand` diisi dengan label **"Unknown"**.
- Kolom numerik seperti `RAM`, `main camera`, dan `battery size` diisi menggunakan **nilai median** untuk meminimalkan pengaruh outlier.

#### 3. Konversi Tipe Data
- Kolom `price` dikonversi menjadi tipe data **float** agar sesuai untuk analisis numerik dan pemodelan.

#### 4. Standarisasi Teks
- Nilai pada kolom `model` dan `brand` telah dinormalisasi dengan:
  - Mengubah seluruh huruf menjadi **lowercase**
  - Menghapus **spasi di awal dan akhir**
- Langkah ini penting untuk menghindari duplikasi tidak langsung akibat perbedaan penulisan teks.

### Data Preparation Users_df
#### 1. Hapus Duplikasi
- Baris duplikat berdasarkan `user_id` telah dihapus.
- Hal ini untuk memastikan bahwa setiap pengguna hanya tercatat sekali dalam dataset.

#### 2. Penanganan Missing Value
- Kolom `gender` yang kosong diisi dengan label **"Unknown"**.
- Kolom `age` yang kosong diisi dengan **median usia** untuk menjaga distribusi data tetap wajar dan mengurangi bias.
- Kolom `occupation_cleaned` (hasil normalisasi dari `occupation`) diisi dengan **"unknown"** jika kosong.

#### 3. Validasi Nilai Usia
- Data difilter agar hanya menyertakan pengguna dengan rentang usia **antara 10 hingga 90 tahun**.
- Tujuan langkah ini adalah untuk menghapus outlier yang tidak realistis (misalnya usia <10 atau >90).


### Data Preparation Ratings_df
#### 1. Pembersihan Nilai Outlier
- Hanya rating dengan rentang **1 hingga 10** yang disertakan.
- Nilai rating di luar batas ini (misalnya 18) dianggap **outlier** dan telah dihapus untuk menjaga integritas data.

#### 2. Penghapusan Duplikasi User–Produk
- Duplikasi pada kombinasi `user_id` dan `cellphone_id` telah dihapus.
- Hal ini memastikan bahwa setiap pengguna hanya memberikan **satu rating unik per produk**.



Kemudian setelah dataset ditangani, masuk ke tahap:
### Sinkronisasi Antar Dataset
Sinkronisasi antar dataset adalah proses penting dalam data preprocessing untuk memastikan bahwa data yang digunakan dalam analisis atau pelatihan model benar-benar konsisten, bersih, dan relevan.
#### 1. Filter Data Valid
- Data `ratings_df` disaring agar hanya memuat:
  - `user_id` yang terdapat dalam `users_df`
  - `cellphone_id` yang terdapat dalam `products_df`
- Langkah ini memastikan bahwa semua interaksi hanya melibatkan pengguna dan produk yang **valid dan terdaftar**.

#### 2. Penggabungan Dataset
- Dataset `ratings_df`, `users_df`, dan `products_df` telah digabung:
  - Pertama berdasarkan `user_id`
  - Kemudian berdasarkan `cellphone_id`
- Hasil penggabungan disimpan dalam `merged_df` untuk **analisis lanjutan** atau **pelatihan model rekomendasi**.

### Dataset Gabungan (`merged_df`)

Dataset ini merupakan hasil penggabungan dari tiga sumber data utama: `users_df`, `products_df`, dan `ratings_df`.

#### Struktur Kolom:
| Kolom               | Keterangan |
|---------------------|------------|
| `user_id`           | ID unik pengguna |
| `cellphone_id`      | ID unik produk/handphone |
| `rating`            | Nilai rating yang diberikan pengguna (1–10) |
| `age`               | Usia pengguna |
| `gender`            | Jenis kelamin pengguna |
| `occupation`        | Pekerjaan asli pengguna |
| `occupation_cleaned`| Pekerjaan yang telah dinormalisasi |
| `brand`             | Merek handphone |
| `model`             | Model handphone |
| `operating system`  | Sistem operasi handphone |
| `internal memory`   | Kapasitas memori internal (GB) |
| `RAM`               | Kapasitas RAM (GB) |
| `performance`       | Skor performa handphone |
| `main camera`       | Resolusi kamera utama (MP) |
| `selfie camera`     | Resolusi kamera depan (MP) |
| `battery size`      | Kapasitas baterai (mAh) |
| `screen size`       | Ukuran layar (inci) |
| `weight`            | Berat handphone (gram) |
| `price`             | Harga (dalam satuan mata uang lokal) |
| `release date`      | Tanggal rilis handphone |

#### Keterangan:
- Dataset ini siap digunakan untuk **sistem rekomendasi**.
- Setiap baris mewakili satu interaksi pengguna terhadap satu produk.
- Kolom-kolom dari `users_df` dan `products_df` telah berhasil ditransfer ke dalam satu frame yang seragam.

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
