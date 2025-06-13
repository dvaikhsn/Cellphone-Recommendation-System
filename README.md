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
```
print("\nList variabel products_df:")
products_df
```
Dari Dataset Products_df, terdapat 14 kolom dan 33 baris yang berisi informasi mengenai spesifikasi berbagai produk handphone:
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
  
Cek missing Value Products_df
```
#Cek missing value products_df
products_df.isnull().sum()
```
  ![image](https://github.com/user-attachments/assets/3344a3a0-afd6-45ef-94e2-bfca3d2ee940)

Cek data duplikat Value Products_df
```
#Cek Data Duplicate value products_df
products_df.duplicated().sum()
```
`0`

Setelah di cek, dataset `products_df` tidak terdapat nilai yang hilang dan duplikat, sehingga proses analisis dapat langsung dilakukan tanpa perlu menangani data kosong atau null.  

---
#### **Dataset ratings_df**
```
print("List variabel ratings_df:")
ratings_df
```
Dari dataset `ratings_df`, terdapat **3 kolom** dan **990 baris** yang berisi informasi mengenai:

1. **`user_id`** – ID unik dari masing-masing pengguna yang memberikan rating.
2. **`cellphone_id`** – ID unik dari masing-masing produk handphone yang diberi rating.
3. **`rating`** – Nilai penilaian yang diberikan oleh pengguna terhadap produk, dalam skala 1–10.
   
    ![image](https://github.com/user-attachments/assets/71d9b122-7def-42f9-a134-e1c2abc320b7)

Cek missing Value ratings_df
```
#Cek missing value users_df
users_df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/9ba8f60d-93ae-4b50-9a37-baab5bb96a45)

Cek data duplikat Value ratings_df
```
#Cek Data Duplicate value ratings_df
ratings_df.duplicated().sum()
```
`0`

Setelah di cek, dataset `ratings_df` tidak terdapat nilai yang hilang dan duplikat, sehingga proses analisis dapat langsung dilakukan tanpa perlu menangani data kosong atau null. 

---
#### **Dataset users_df** 
```
print("\nList variabel users_df:")
users_df
```
Dari dataset `users_df`, terdapat **4 kolom** dan **99 baris** yang berisi informasi mengenai:

1. **`user_id`** – ID unik dari masing-masing pengguna.
2. **`age`** – Usia pengguna.
3. **`gender`** – Jenis kelamin pengguna (misalnya, 'M' untuk male dan 'F' untuk female).
4. **`occupation`** – Pekerjaan pengguna (dalam format kategori atau teks).

   ![image](https://github.com/user-attachments/assets/0663f357-443c-465d-8bdf-9212a68ad35a)

Cek missing Value users_df
```
#Cek missing value users_df
users_df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a64ead72-a579-4e2d-98ff-a06ea6b8fae0)

Cek data duplikat Value users_df
```
#Cek Data Duplicate value ratings_df
users_df.duplicated().sum()
```
`0`

Setelah di cek, dataset `users_df` hanya memiliki 1 nilai yang hilang pada kolom `occupation` dan tidak terdapat nilai yang duplikat.

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
- Mengubah ke huruf kecil
- Menghilangkan spasi di awal/akhir
- Menghapus karakter non-huruf 

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
- Membuang data dengan rating di luar rentang 1–10. Sehingga dihasilkan seperti pada visualisasi di atas. Visual ini terlihat bahwa:
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

### Penanganan Data Duplikat
Untuk memastikan akurasi analisis dan efisiensi komputasi, data duplikat dihapus menggunakan fungsi drop_duplicates() dari pandas. Duplikasi data dapat menyebabkan bias karena nilai-nilai yang sama dihitung lebih dari sekali, serta memperlambat proses pelatihan model.

### Penanganan Missing Value
Data yang hilang ditangani dengan beberapa metode, seperti mengisi nilai berdasarkan modus atau melakukan interpolasi, tergantung pada konteks variabel. Penanganan ini penting agar tidak mengganggu proses analisis dan menjaga performa model tetap optimal.

### Penggabungan Dataset
Dataset ratings_df, users_df, dan products_df digabung menggunakan fungsi merge() dari pandas. Penggabungan ini diperlukan untuk membangun sistem rekomendasi berbasis preferensi pengguna, karena masing-masing dataset menyediakan informasi pelengkap satu sama lain seperti spesifikasi produk, identitas pengguna, dan rating yang diberikan.

### Transformasi Fitur Teks dengan TF-IDF
Fitur teks seperti nama model handphone diubah menjadi representasi numerik menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency). Representasi ini memungkinkan sistem mengenali kemiripan antar produk berdasarkan deskripsi teksnya.

Langkah-langkah berikut dilakukan untuk menyiapkan data dari masing-masing dataset sebelum proses pemodelan. Semua teknik disusun sesuai urutan eksekusi di notebook:

### Data Preparation Products_df

Karena tidak ada data yang hilang dan duplikat pada dataset products_df, sehingga preparation yang dilakukan sebagai berikut:

#### 1. Konversi Tipe Data
- Kolom `price` dikonversi menjadi tipe data **float** agar sesuai untuk analisis numerik dan pemodelan.

#### 2. Standarisasi Teks
- Nilai pada kolom `model` dan `brand` telah dinormalisasi dengan:
- Mengubah seluruh huruf menjadi **lowercase**
- Menghapus **spasi di awal dan akhir**
- Langkah ini penting untuk menghindari duplikasi tidak langsung akibat perbedaan penulisan teks.

### Data Preparation Users_df

#### 1. Penanganan Missing Value
- Kolom `occupation_cleaned` (hasil normalisasi teks dari `occupation` berdasarkan huruf kecil dan besarnya).
- Karena pada dataset users_df ini terdapat `1` data yang hilang, maka dari itu `occupation_cleaned` hasil normalisasi ini  diisi dengan **"unknown"** jika datanya kosong.

#### 2. Validasi Nilai Usia
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

### Persiapan

---

## Modeling
Model rekomendasi dibangun dengan dua pendekatan utama:
### **Content Based Filtering**
Pendekatan ini menggunakan teknik **Content-Based Filtering** untuk merekomendasikan produk (handphone) berdasarkan kemiripan fitur antar produk.

1. **Ekstraksi Fitur Produk**
```
# Gabungkan fitur penting menjadi satu string teks
products_df['features'] = (
    products_df['brand'] + ' ' +
    products_df['model'] + ' ' +
    products_df['RAM'].astype(str) + 'GB RAM ' +
    products_df['battery size'].astype(str) + 'mAh '
)
```
   Fitur-fitur penting dari setiap produk digabung menjadi satu string teks. Fitur yang digunakan:
   - `brand`
   - `model`
   - `RAM`
   - `battery size`

2. **Representasi Teks (TF-IDF)**
```
# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(products_df['features'])
```
- Menggunakan `TfidfVectorizer` untuk mengubah teks menjadi representasi numerik berdasarkan frekuensi term yang muncul.
- TF-IDF membantu menekankan kata-kata unik yang membedakan satu produk dari lainnya.

3. **Penghitungan Kemiripan Produk**
```
# Matriks kesamaan antar produk
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```
- Menggunakan **cosine similarity** untuk mengukur kemiripan antar vektor fitur produk.
- Semakin tinggi nilai cosine similarity (mendekati 1), semakin mirip dua produk tersebut.

4. **Fungsi Rekomendasi**
```
def recommend_by_product(product_name, top_n=5):
    idx = indices.get(product_name)
    if idx is None:
        return "Produk tidak ditemukan."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return products_df.iloc[product_indices][['model', 'brand', 'price']]
```
Fungsi `recommend_by_product()` menerima input nama produk (misalnya `"galaxy s22"`) dan mengembalikan `top_n` produk paling mirip berdasarkan nilai kemiripan tertinggi.

### **Collaborative Filtering**
Pendekatan ini menggunakan teknik **Collaborative Filtering** dengan algoritma **Singular Value Decomposition (SVD)** dari library `Surprise`.

1. **Persiapan Dataset**
```
# Buat dataset untuk Surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_df[['user_id', 'cellphone_id', 'rating']], reader)
```
   - Data digunakan: `ratings_df` yang berisi `user_id`, `cellphone_id`, dan `rating`.
   - Library `Surprise` digunakan untuk membangun model SVD.

2. **Pembagian Data**
```
# Split train-test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
```
   - Dataset dibagi menjadi data pelatihan (80%) dan pengujian (20%) menggunakan `train_test_split`.

4. **Pelatihan Model**
```
# Latih model SVD
svd = SVD()
svd.fit(trainset)
```
   - Model SVD dilatih menggunakan data pelatihan.

6. **Evaluasi Model**
```
# Evaluasi
predictions = svd.test(testset)
print("\nRMSE dari Collaborative Filtering (SVD):")
rmse(predictions)
```
   - Evaluasi dilakukan menggunakan metrik **RMSE (Root Mean Squared Error)** pada data pengujian.
   - Semakin kecil nilai RMSE, semakin baik kualitas prediksi rating.

##  Evaluation
### **Content-Based Filtering**

Evaluasi ini bertujuan untuk menganalisis hasil dari sistem rekomendasi berbasis **Content-Based Filtering**, yaitu dengan membandingkan kesamaan fitur antar produk.
---

## 1. Metode Evaluasi
```
# Feature matrix
features_cb = products_df[['RAM', 'main camera', 'battery size', 'price']].copy()
features_cb_scaled = MinMaxScaler().fit_transform(features_cb)
cos_sim = cosine_similarity(features_cb_scaled)
```
- Model ini menggunakan **kemiripan atribut produk** seperti:
  - **RAM**
  - **Main Camera**
  - **Battery Size**
  - **Price**
  - Semua fitur dinormalisasi menggunakan **MinMaxScaler** agar memiliki skala yang seimbang.
  - Kemiripan antar produk dihitung menggunakan **cosine similarity**.
  - Produk yang paling mirip dengan produk target akan ditampilkan berdasarkan skor similarity tertinggi.

---

## 2. Contoh Rekomendasi: Produk Mirip 'Galaxy S22'
```
# Lihat produk yang mirip dengan Galaxy S22
target_model = 'galaxy s22'

# Ambil index dari produk target
idx = products_df[products_df['model'] == target_model].index[0]
sim_scores = list(enumerate(cos_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
top_similar_idx = [i for i, score in sim_scores[1:6]]

cb_results = products_df.iloc[top_similar_idx][['brand', 'model']]
cb_results['similarity'] = [sim_scores[i][1] for i in range(1, 6)]

print(f"\nTop 5 Produk Mirip '{target_model.title()}' (Content-Based Filtering):")
for _, row in cb_results.iterrows():
    print(f"- {row['brand'].title()} {row['model'].title()} (Similarity: {row['similarity']:.2f})")
```
Berikut adalah **5 produk teratas** yang memiliki kesamaan fitur tertinggi dengan Galaxy S22:


```
Top 5 Produk Mirip 'Galaxy S22' (Content-Based Filtering):
- Asus Zenfone 8 (Similarity: 0.99)
- Google Pixel 6 Pro (Similarity: 0.98)
- Xiaomi Poco F4 (Similarity: 0.98)
- Oppo Find X5 Pro (Similarity: 0.98)
- Xiaomi 12 Pro (Similarity: 0.98)
```
## 3. Visualisasi

![image](https://github.com/user-attachments/assets/5c546609-1dda-4589-ba29-d03bb753f377)

## 3. Interpretasi

- Produk-produk di atas memiliki spesifikasi teknis yang sangat mirip dengan Galaxy S22, sehingga cocok sebagai alternatif.
- Kemiripan diukur bukan dari merek, tetapi dari **fitur dan performa** teknis.
- Semua produk memiliki **skor kemiripan mendekati 1.0**, menunjukkan bahwa fitur-fitur teknisnya sangat serupa.
- Produk seperti **Zenfone 8** dan **Pixel 6 Pro** menjadi alternatif utama.
- Ini membantu pengguna menemukan opsi dengan **fitur sebanding**, meski dari brand berbeda atau harga lebih terjangkau.

---

### **Collaborative Filtering**
Evaluasi ini bertujuan untuk mengukur kinerja model rekomendasi berbasis **Collaborative Filtering dengan algoritma SVD** menggunakan data rating user terhadap produk smartphone.

---

##  1. Metode Evaluasi
```
# 1. Hitung RMSE (tanpa squared=False)
y_true = [true_r for (_, _, true_r) in testset]
y_est = [pred.est for pred in predictions]

mse_score = mean_squared_error(y_true, y_est)
rmse_score = np.sqrt(mse_score)
```
- **RMSE (Root Mean Squared Error)** digunakan sebagai metrik utama evaluasi.
- RMSE mengukur **selisih rata-rata antara rating aktual dan rating yang diprediksi**.
- Semakin rendah nilai RMSE, semakin akurat prediksi yang dihasilkan oleh model.

###  Hasil RMSE: RMSE Model Collaborative Filtering (SVD): 2.1936

  Insight:
- Nilai RMSE 2.19 pada skala rating 1–10 tergolong cukup baik.
- Ini menunjukkan bahwa prediksi model relatif dekat dengan rating aktual yang diberikan pengguna.
- Namun, masih ada ruang untuk peningkatan, misalnya dengan optimasi parameter atau kombinasi model (hybrid).

---

##  2. Contoh Rekomendasi untuk User ID = 1

Berikut adalah **5 rekomendasi teratas** yang dihasilkan model untuk user dengan ID 1:
```
Top 5 Rekomendasi (Collaborative Filtering) untuk User 1:
- iphone 13 pro (Prediksi Rating: 8.45)
- moto g power (2022) (Prediksi Rating: 6.63)
```

## 3. Visualisasi

![image](https://github.com/user-attachments/assets/6e9863fe-f184-41f1-96c1-1b46eaae0b21)

  Insight:
- Model merekomendasikan produk flagship dan mid-range, menandakan model menangkap preferensi berdasarkan pola rating user lain yang serupa.
- Produk dengan prediksi rating tinggi cenderung berasal dari brand populer seperti Apple dan Motorola.
- Rekomendasi bisa menjadi dasar personalisasi yang baik untuk user baru maupun aktif.
- iPhone 13 Pro memiliki rating prediksi tertinggi (~8.7) → kemungkinan besar disukai oleh User 1.
- Moto G Power (2022) juga direkomendasikan tapi dengan skor lebih rendah (~6.8).


# **Evaluasi Terhadap Business Understanding**
## ✅ Problem Statements
1.  Bagaimana cara membantu pengguna menemukan smartphone yang sesuai dengan kebutuhan mereka di tengah banyaknya pilihan yang tersedia?
  `Melalui pendekatan Content-Based Filtering, sistem mampu merekomendasikan produk serupa berdasarkan spesifikasi teknis, membantu pengguna yang memiliki preferensi fitur tertentu.`
  
2. Bagaimana sistem dapat memberikan rekomendasi yang relevan agar pengguna tidak kebingungan dalam mengambil keputusan pembelian?
  `Dengan Collaborative Filtering (SVD), sistem berhasil memberikan rekomendasi personal yang relevan berdasarkan pola rating pengguna lain, mempermudah pengguna dalam memilih.`

## ✅ Goals
1. Membangun sistem rekomendasi smartphone berbasis konten dan kolaborasi.
   `Kedua pendekatan telah diimplementasikan dengan baik, masing-masing menangani aspek berbeda dari sistem rekomendasi.`
   
2. Memberikan rekomendasi yang akurat untuk meningkatkan kepuasan pengguna.
   `Tercapai sebagian besar. Hasil evaluasi menunjukkan bahwa Content-Based Filtering memberikan alternatif produk yang sangat mirip, dan Collaborative Filtering memiliki RMSE sebesar 2.19, yang tergolong cukup baik dalam skala rating 1–10.`

## ✅ Solution Approach
1. **Content-Based Filtering**: Menggunakan atribut produk (brand, model, RAM, kamera, baterai, dan harga) untuk mengukur kesamaan antar produk.
   `Berhasil diterapkan. Sistem mampu menghitung kemiripan antar produk menggunakan TF-IDF dan cosine similarity. Produk-produk seperti Asus Zenfone 8 dan Pixel 6 Pro berhasil direkomendasikan sebagai alternatif mirip Galaxy S22.`
  
2. **Collaborative Filtering (SVD)**: Menggunakan data rating dari pengguna terhadap produk untuk merekomendasikan smartphone yang disukai oleh pengguna dengan preferensi serupa.
   `Berhasil diterapkan. Sistem menggunakan data rating pengguna dan model SVD untuk menghasilkan rekomendasi personal. Evaluasi menggunakan RMSE mengindikasikan performa model yang cukup baik untuk prediksi rating.`
   
---


# KESIMPULAN
## 1. Content-Based Filtering

- Sistem ini membandingkan produk berdasarkan **kemiripan fitur teknis** seperti RAM, kamera utama, kapasitas baterai, dan harga.
- Menggunakan teknik **TF-IDF** dan **cosine similarity** untuk mengukur kesamaan antar produk.
- Contoh hasil: Produk seperti *Zenfone 8*, *Pixel 6 Pro*, dan *Xiaomi 12 Pro* muncul sebagai alternatif yang sangat mirip dengan *Galaxy S22*.
- **Visualisasi bar chart** memperkuat rekomendasi dengan menunjukkan skor kemiripan mendekati 1.0.

📌 **Kelebihan**:
- Akurat untuk produk yang belum banyak diulas user.
- Sangat baik dalam memberi alternatif serupa dari brand berbeda.

---

## 2. Collaborative Filtering (SVD)

- Model ini belajar dari perilaku pengguna (user-product rating) untuk memprediksi produk yang mungkin disukai user lainnya.
- Menggunakan algoritma **Singular Value Decomposition (SVD)** dari library `Surprise`.
- Evaluasi dengan **RMSE = 2.19** (dari skala 1–10), menunjukkan prediksi rating yang cukup baik.
- Contoh output: User ID 1 direkomendasikan produk seperti *iPhone 13 Pro* dengan estimasi rating 8.45.

📌 **Kelebihan**:
- Personalisasi berdasarkan preferensi pengguna.
- Cocok untuk memberikan saran yang disesuaikan bagi pengguna aktif.

---

## 3. Perbandingan & Insight

| Aspek                        | Content-Based Filtering                        | Collaborative Filtering (SVD)               |
|-----------------------------|------------------------------------------------|---------------------------------------------|
| Data yang dibutuhkan        | Fitur produk                                   | Rating dari pengguna                         |
| Kemampuan personalisasi     | Terbatas                                       | Tinggi                                       |
| Cold-start (produk baru)    | Bisa direkomendasikan                          | Tidak bisa tanpa rating                      |
| Cold-start (user baru)      | Tidak relevan (tidak butuh user)               | Sulit (butuh interaksi pengguna)            |
| Hasil evaluasi              | Top 5 produk mirip akurat                      | RMSE: 2.19 (cukup baik)                      |

---

## Kesimpulan Sistem

Model sistem rekomendasi ini berhasil menunjukkan performa yang cukup baik dalam dua pendekatan berbeda. **Content-Based Filtering** memberikan alternatif produk yang mirip secara fitur, sedangkan **Collaborative Filtering (SVD)** memberi rekomendasi personal berdasarkan perilaku pengguna lain.

