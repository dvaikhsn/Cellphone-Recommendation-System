# -*- coding: utf-8 -*-
"""Dava_Ikhsan_R_Machine_Learning_Terapan_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IEzD9P4VYBvwNo4bpJePrpzjtYj92AEs

# **Cellphone Recommendation System**

## **Nama: Dava Ikhsan Reyvan**

## **Email: dvaikhsn@gmail.com**

## **ID Dicoding: dvikhsn**
"""

!pip install numpy==1.23.5 scikit-surprise --no-binary scikit-surprise

"""### **Library yang dipakai**"""

import seaborn as sns
import pandas as pd
import requests
import shutil
import numpy as np
import os
import zipfile
from PIL import Image, UnidentifiedImageError
from matplotlib import pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

"""# **Data Understanding**

## **Load dataset**
"""

from google.colab import drive
drive.mount ('/content/drive')

ratings_df = pd.read_csv("/content/drive/My Drive/Machine Learning Terapan #2/cellphones ratings.csv")
users_df = pd.read_csv("/content/drive/My Drive/Machine Learning Terapan #2/cellphones users.csv")
products_df = pd.read_csv("/content/drive/My Drive/Machine Learning Terapan #2/cellphones data.csv")

"""## **Dataset Produk**"""

print("\nList variabel products_df:")
products_df

"""### **Insight:**

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

## **Dataset Rating**
"""

print("List variabel ratings_df:")
ratings_df

"""### **Insight:**

Dari dataset `ratings_df`, terdapat **3 kolom** dan **990 baris** yang berisi informasi mengenai:

1. **`user_id`** – ID unik dari masing-masing pengguna yang memberikan rating.
2. **`cellphone_id`** – ID unik dari masing-masing produk handphone yang diberi rating.
3. **`rating`** – Nilai penilaian yang diberikan oleh pengguna terhadap produk, dalam skala 1–10.

## **Dataset User**
"""

print("\nList variabel users_df:")
users_df

"""### **Insight:**

Dari dataset `users_df`, terdapat **4 kolom** dan **99 baris** yang berisi informasi mengenai:

1. **`user_id`** – ID unik dari masing-masing pengguna.
2. **`age`** – Usia pengguna.
3. **`gender`** – Jenis kelamin pengguna (misalnya, 'M' untuk male dan 'F' untuk female).
4. **`occupation`** – Pekerjaan pengguna (dalam format kategori atau teks).

## **Assesing Data**

# **Cek Missing Value**

## **Dataset Produk**
"""

#Cek missing value products_df
products_df.isnull().sum()

"""### **Insight:**

Dari dataset `products_df` tidak terdapat nilai yang hilang, sehingga proses analisis dapat langsung dilakukan tanpa perlu menangani data kosong atau null.

## **Dataset User**
"""

#Cek missing value users_df
users_df.isnull().sum()

"""### **Insight:**

Dari dataset `users_df` hanya memiliki 1 nilai yang hilang pada kolom `occupation`

## **Dataset Rating**
"""

#Cek missing value ratings_df
ratings_df.isnull().sum()

"""### **Insight:**

Dari dataset `ratings_df` tidak terdapat nilai yang hilang, sehingga proses analisis dapat langsung dilakukan tanpa perlu menangani data kosong atau null.

## **Cek Data Duplicate**
"""

#Cek Data Duplicate value products_df
products_df.duplicated().sum()

#Cek Data Duplicate value users_df
users_df.duplicated().sum()

#Cek Data Duplicate value ratings_df
ratings_df.duplicated().sum()

"""### **Insight:**

Dari ketiga dataset (`ratings_df`, `users_df`, dan `products_df`), **tidak ada data yang duplikat**

# **Exploratory Data Analysis & Visualisasi Data**

# **Dataset Produk**
"""

products_df.info()

"""### **Insight EDA dari Dataset `products_df`**

Dataset `products_df` berisi **33 baris** dan **14 kolom** yang menggambarkan spesifikasi berbagai model handphone. Berikut adalah beberapa insight awal berdasarkan analisis eksploratif:

---

### **Struktur Data**

- Semua kolom memiliki **nilai lengkap (non-null)** tanpa missing data.
- Terdapat 8 kolom bertipe numerik (int64 dan float64), seperti `internal memory`, `RAM`, `performance`, `main camera`, `selfie camera`, `battery size`, `screen size`, `weight`, dan `price`.
- Terdapat 4 kolom bertipe objek, yaitu `brand`, `model`, `operating system`, dan `release date`.
- Dataset relatif kecil (33 baris), sehingga analisis bisa lebih mendalam pada setiap fitur.
"""

products_df.describe(include='all')

# Distribusi Harga Handphone
plt.figure(figsize=(10,5))
sns.histplot(products_df['price'].dropna(), bins=30, kde=True, )
plt.title('Distribusi Harga Handphone')
plt.tight_layout()
plt.show()

"""### **Insight:**

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
"""

# Korelasi antara fitur numerik
numerical = products_df[['RAM', 'main camera', 'battery size', 'price']]
sns.heatmap(numerical.corr(), annot=True, cmap='Blues')
plt.title("Korelasi Fitur Numerik")
plt.show()

"""### **Insight:**

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

# **Dataset User**
"""

print("Informasi Dataset User:")
print(users_df.info())

"""## **Insight:**
- Jumlah data: 99 baris
- Jumlah fitur: 4 kolom
- Tipe data:
  - `user_id` (int64): ID unik tiap pengguna
  - `age` (int64): Usia pengguna
  - `gender` (object): Jenis kelamin pengguna
  - `occupation` (object): Pekerjaan pengguna
  - Kolom `occupation` memiliki **1 nilai yang hilang** (NaN).

"""

print("\nStatistik Deskriptif:")
users_df.describe(include='all')

"""### Insight:

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

"""

# Visualisasi distribusi usia pengguna
plt.figure(figsize=(8, 5))
sns.histplot(users_df['age'].dropna(), bins=20, kde=True, color='teal')
plt.title("Distribusi Usia Pengguna")
plt.xlabel("Usia")
plt.ylabel("Jumlah Pengguna")
plt.show()

"""### Insight:

- Mayoritas pengguna berusia **25–35 tahun**.
- Distribusi condong ke kanan (right-skewed), artinya lebih banyak pengguna muda dibandingkan yang lebih tua.
- Sedikit pengguna yang berusia di atas 50 tahun.
"""

# Visualisasi jumlah pengguna berdasarkan jenis kelamin
if 'gender' in users_df.columns:
    gender_clean = users_df[users_df['gender'].isin(['Male', 'Female'])]

    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', data=gender_clean, palette='pastel')
    plt.title("Distribusi Pengguna berdasarkan Jenis Kelamin")
    plt.xlabel("Jenis Kelamin")
    plt.ylabel("Jumlah Pengguna")
    plt.grid(axis='y')
    plt.show()

"""### **Insight:**

- Jumlah pengguna pria dan wanita hampir seimbang.
- **Pria sedikit lebih banyak** dibanding wanita (50 vs 46 pengguna).

"""

# Visualisasi jumlah pengguna berdasarkan pekerjaan dengan horizontal bar chart
if 'occupation' in users_df.columns:
    plt.figure(figsize=(10, 12))
    occupation_counts = users_df['occupation'].value_counts()

    sns.barplot(y=occupation_counts.index, x=occupation_counts.values, palette='coolwarm')
    plt.title("Distribusi Pengguna berdasarkan Pekerjaan")
    plt.xlabel("Jumlah Pengguna")
    plt.ylabel("Pekerjaan")
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

"""### **Insight:**

- Sebagian besar pengguna dalam dataset berasal dari latar belakang teknologi, dengan pekerjaan **Information Technology** menjadi yang paling umum, terlihat ada  **10 pengguna**. Hal ini menunjukkan potensi minat teknologi yang kuat di antara pengguna.

- Pekerjaan pengguna juga bervariasi, karena terdapat **56 kategori pekerjaan**
- Terdapat beberapa entri yang duplikat penulisan seperti `manager`, `Manager`, dan `Manager` → perlu **normalisasi teks**.

"""

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

"""### **Insight**
- `.str.lower()`: Menyatukan kapitalisasi.

- `.str.strip()`: Menghapus spasi berlebih.

- `.str.replace(...)`: Menghapus karakter seperti angka, tanda baca, dll

- `value_counts()`: Untuk melihat distribusi setelah normalisasi.
"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 12))
sns.countplot(
    data=users_df,
    y='occupation_cleaned',
    order=users_df['occupation_cleaned'].value_counts().index,
    palette='coolwarm'
)
plt.title('Distribusi Pekerjaan Pengguna (Setelah Normalisasi)')
plt.xlabel('Jumlah Pengguna')
plt.ylabel('Pekerjaan')
plt.tight_layout()
plt.show()

"""### Distribusi Pekerjaan Pengguna (Setelah Normalisasi)

📊 **Normalisasi Teks**  
Untuk menghindari duplikasi akibat perbedaan kapitalisasi dan format, dilakukan normalisasi teks pada kolom `occupation`:
- Lowercase seluruh nilai
- Menghapus spasi berlebih
- Menghapus karakter non-alfabet

```python
users_df['occupation_cleaned'] = (
    users_df['occupation']
    .str.lower()
    .str.strip()
    .str.replace(r'[^a-z\s]', '', regex=True)
)

# **Dataset Rating**
"""

ratings_df.info()

"""## **Insight:**
- Jumlah data: 990 baris
- Jumlah fitur: 3 kolom
- Tipe data:
  - `user_id` (int64): ID pengguna
  - `cellphone_id` (int64): ID handphone
  - `rating` (int64): Nilai rating dari pengguna

> Semua kolom tidak ada yang memiliki nilai kosong.

"""

ratings_df.describe(include='all')

"""### Insight:

- **Rata-rata rating:** 6.7  
- **Median rating:** 7  
- **Rating minimum:** 1  
- **Rating maksimum:** 18  
- **Sebaran rating (IQR):** 5 hingga 9

> Catatan: Terdapat outlier pada rating maksimum (18), dari skala 1–10.
"""

# Distribusi Nilai Rating

# Buang data dengan rating di luar rentang 1–10
ratings_df = ratings_df[(ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 10)]

plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=ratings_df, palette='Set2')
plt.title("Distribusi Nilai Rating")
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.grid(axis='y')
plt.show()

"""### **Insight:**

- Rating paling sering diberikan yaitu 8, terlhat hampir 200 kali.
- Sebagian besar pengguna memberikan **rating antara 6–10**, menunjukkan kecenderungan penilaian positif.

"""

# Jumlah rating per produk
rating_counts = ratings_df['cellphone_id'].value_counts()
top_rated = rating_counts.head(10).index

plt.figure(figsize=(10,5))
sns.barplot(x=products_df[products_df['cellphone_id'].isin(top_rated)]['model'],
            y=rating_counts[top_rated].values)
plt.xticks(rotation=45)
plt.title("10 Handphone dengan Rating Terbanyak")
plt.xlabel("Model")
plt.ylabel("Jumlah Rating")
plt.show()

"""### **Insight:**

- **iPhone 13 Pro Max** menjadi model yang paling banyak mendapatkan rating, yaitu lebih dari **40 rating**.
- Disusul oleh **Galaxy A32**, **Galaxy S22 Ultra**, dan **Galaxy Z Flip 3** yang masing-masing memiliki hampir **40 rating**.
- Secara keseluruhan, model dari **Apple**, **Samsung**, **Xiaomi**, dan **Motorola** mendominasi daftar 10 besar.
- Jumlah rating tinggi bisa mengindikasikan popularitas atau banyaknya pengguna yang terlibat dengan produk tersebut.

# **Data Preparation**

# **Dataset Produk**
"""

# PRODUCTS DATA PREPARATION

# 1. Konversi tipe data jika perlu
products_df['price'] = products_df['price'].astype(float)

# 2. Standarisasi teks
products_df['model'] = products_df['model'].str.lower().str.strip()
products_df['brand'] = products_df['brand'].str.lower().str.strip()

"""###  Data Preparation - Produk

Karena tidak ada data yang hilang dan duplikat pada dataset products_df, sehingga preparation yang dilakukan sebagai berikut:

#### 1. Konversi Tipe Data
- Kolom `price` dikonversi menjadi tipe data **float** agar sesuai untuk analisis numerik dan pemodelan.

#### 2. Standarisasi Teks
- Nilai pada kolom `model` dan `brand` telah dinormalisasi dengan:
  - Mengubah seluruh huruf menjadi **lowercase**
  - Menghapus **spasi di awal dan akhir**
- Langkah ini penting untuk menghindari duplikasi tidak langsung akibat perbedaan penulisan teks.

# **Dataset User**
"""

# USERS DATA PREPARATION

# 1. Tangani nilai kosong
users_df['occupation_cleaned'] = users_df['occupation_cleaned'].fillna('unknown')

# 2. Validasi nilai usia
users_df = users_df[(users_df['age'] >= 10) & (users_df['age'] <= 90)]  # Range usia wajar

"""### Data Preparation - Users

#### 1. Penanganan Missing Value
- Kolom `occupation_cleaned` (hasil normalisasi teks dari `occupation` berdasarkan huruf kecil dan besarnya).
Karena pada dataset users_df ini terdapat `1` data yang hilang, maka dari itu `occupation_cleaned` hasil normalisasi ini  diisi dengan **"unknown"** jika datanya kosong.

#### 2. Validasi Nilai Usia
- Data difilter agar hanya menyertakan pengguna dengan rentang usia **antara 10 hingga 90 tahun**.
- Tujuan langkah ini adalah untuk menghapus outlier yang tidak realistis (misalnya usia <10 atau >90).

# **Dataset Rating**
"""

#  RATINGS DATA PREPARATION

# 1. Sudah dibersihkan dari nilai outlier (rating > 10)
ratings_df = ratings_df[(ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 10)]

# 2. Cek duplikasi rating user–produk
ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'cellphone_id'])

"""### Data Preparation - Ratings

#### 1. Pembersihan Nilai Outlier
- Hanya rating dengan rentang **1 hingga 10** yang disertakan.
- Nilai rating di luar batas ini (misalnya 18) dianggap **outlier** dan telah dihapus untuk menjaga integritas data.

#### 2. Penghapusan Duplikasi User–Produk
- Duplikasi pada kombinasi `user_id` dan `cellphone_id` telah dihapus.
- Hal ini memastikan bahwa setiap pengguna hanya memberikan **satu rating unik per produk**.

# **Sinkronisasi Antar Dataset**

Untuk memastikan data konsisten, hanya interaksi yang valid (produk dan pengguna ada di masing-masing tabel) yang dipertahankan:
"""

ratings_df = ratings_df[
    (ratings_df['user_id'].isin(users_df['user_id'])) &
    (ratings_df['cellphone_id'].isin(products_df['cellphone_id']))
]

"""digunakan untuk memfilter DataFrame ratings_df agar hanya menyisakan data rating yang valid, yaitu hanya baris-baris dengan user_id yang terdapat dalam DataFrame users_df dan cellphone_id yang terdapat dalam DataFrame products_df. Dengan menggunakan fungsi .isin(), proses ini memastikan bahwa hanya pengguna dan produk yang terdaftar secara resmi yang disertakan dalam analisis selanjutnya, sehingga membantu menjaga konsistensi dan kualitas data.

# **Persiapan untuk Content-Based Filtering**

Untuk membangun sistem rekomendasi berbasis konten, kita membuat fitur gabungan berupa teks yang akan diekstraksi menggunakan TF-IDF.
"""

# Pastikan 'features' sudah ada
products_df['features'] = (
    products_df['brand'] + ' ' +
    products_df['model'] + ' ' +
    products_df['RAM'].astype(str) + 'GB RAM'
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products_df['features'])

# Cosine Similarity antar produk
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

"""Kode di atas untuk menghitung kemiripan antar produk ponsel berdasarkan fitur teks yang digabung dari kolom brand, model, dan RAM. Pertama, dibuat kolom features yang menyatukan informasi tersebut menjadi deskripsi singkat setiap produk, misalnya "Samsung Galaxy A12 4GB RAM". Kemudian, deskripsi ini dikonversi menjadi representasi numerik menggunakan teknik TF-IDF (TfidfVectorizer) dengan menghapus stop words bahasa Inggris agar fokus pada kata-kata penting. Hasil vektorisasi ini disimpan dalam tfidf_matrix, yang selanjutnya digunakan untuk menghitung cosine similarity antar produk, sehingga menghasilkan matriks cosine_similarities yang menunjukkan seberapa mirip setiap produk dengan produk lainnya berdasarkan konten teksnya."""

#Ekstraksi Fitur dengan TF-IDF:
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(products_df['features'])

"""Kode ini melakukan ekstraksi fitur dari teks menggunakan TF-IDF (Term Frequency–Inverse Document Frequency), sebuah teknik untuk mengubah data teks menjadi representasi numerik berbobot berdasarkan frekuensi kemunculan kata yang dianggap penting. Dengan TfidfVectorizer(), data teks dari kolom features diubah menjadi matriks TF-IDF (tfidf_matrix), di mana setiap baris mewakili satu produk dan setiap kolom mewakili satu kata unik yang muncul di seluruh deskripsi produk. Nilai dalam matriks mencerminkan pentingnya kata tersebut dalam deskripsi produk tertentu dibandingkan dengan semua produk lainnya."""

# Perhitungan Kemiripan Produk:
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

"""Kode ini menghitung kemiripan antar produk berdasarkan fitur teks yang telah diekstraksi menggunakan TF-IDF. Dengan menggunakan cosine_similarity, setiap pasangan produk dibandingkan berdasarkan sudut (cosine) antara vektor representasi mereka dalam ruang fitur. Hasilnya disimpan dalam matriks cosine_sim, di mana setiap elemen [i][j] menunjukkan tingkat kemiripan antara produk ke-i dan ke-j. Nilai cosine similarity berkisar antara 0 hingga 1, di mana nilai yang lebih tinggi menunjukkan tingkat kemiripan yang lebih besar.

# **Persiapan untuk Collaborative Filtering (SVD)**
"""

#Pemilihan Kolom Penting:
ratings_df = ratings_df[['user_id', 'cellphone_id', 'rating']]

"""Kode ini digunakan untuk memilih kolom-kolom penting dari DataFrame ratings_df, yaitu user_id, cellphone_id, dan rating. Ketiga kolom ini merupakan informasi utama yang diperlukan dalam sistem rekomendasi berbasis rating, di mana user_id mengidentifikasi pengguna, cellphone_id mengidentifikasi produk yang dinilai, dan rating menunjukkan nilai penilaian yang diberikan pengguna terhadap produk tersebut. Dengan menyederhanakan DataFrame hanya pada kolom-kolom ini, proses analisis dan pemodelan dapat dilakukan dengan lebih fokus dan efisien."""

#Format Data untuk Surprise Library:
from surprise import Dataset, Reader

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_df, reader)

"""Kode ini memformat data rating agar sesuai dengan struktur yang dibutuhkan oleh library Surprise, yang digunakan untuk membangun sistem rekomendasi. Objek Reader digunakan untuk menentukan skala rating, dalam hal ini dari 1 hingga 10. Kemudian, Dataset.load_from_df() digunakan untuk mengubah DataFrame ratings_df menjadi format internal Surprise, di mana hanya tiga kolom yang digunakan secara berurutan: user_id, cellphone_id, dan rating. Format ini memungkinkan data untuk diproses lebih lanjut oleh algoritma rekomendasi dalam library Surprise."""

#Split Data untuk Pelatihan dan Pengujian:
from surprise.model_selection import train_test_split as surprise_train_test_split

trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

"""Kode ini membagi data rating ke dalam dua bagian, yaitu data pelatihan (trainset) dan data pengujian (testset), dengan menggunakan fungsi train_test_split dari library Surprise. Proporsi data pengujian ditetapkan sebesar 20% (test_size=0.2), sedangkan sisanya digunakan untuk melatih model rekomendasi. Parameter random_state=42 digunakan agar pembagian data bersifat konsisten setiap kali kode dijalankan. Pembagian ini penting untuk mengevaluasi kinerja model secara objektif dengan membandingkan prediksi terhadap data yang belum pernah dilihat selama pelatihan.

## **Modeling**

# **Content Based Filtering**
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF vectorizer dari kolom 'features' yang sudah disiapkan di data preparation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(products_df['features'])

# Hitung cosine similarity antar produk
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Simpan dataframe dengan index produk
products_df = products_df.reset_index(drop=True)

# Fungsi untuk rekomendasi berbasis konten
def recommend_content_based(cellphone_id, products_df, cosine_sim, top_n=5):
    # Temukan index dari produk berdasarkan cellphone_id
    idx_list = products_df.index[products_df['cellphone_id'] == cellphone_id].tolist()
    if not idx_list:
        return pd.DataFrame()
    idx = idx_list[0]

    # Hitung skor kemiripan
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Ambil produk-produk paling mirip
    similar_indices = [i[0] for i in sim_scores]
    recommended = products_df.iloc[similar_indices][['cellphone_id', 'model', 'brand']].copy()
    recommended['similarity_score'] = [i[1] for i in sim_scores]

    return recommended

# Contoh pemanggilan model
sample_id = products_df.iloc[2]['cellphone_id']
recommendations_cb = recommend_content_based(sample_id, products_df, cosine_sim, top_n=5)
recommendations_cb

"""# **Collaborative Filtering**"""

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split

# Format ulang data ratings ke format Surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_df[['user_id', 'cellphone_id', 'rating']], reader)

# Split data (80% train, 20% test)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Inisialisasi dan latih model SVD
svd_model = SVD()
svd_model.fit(trainset)

# Fungsi untuk mendapatkan top-N rekomendasi bagi user tertentu
def get_top_n_recommendations(model, user_id, products_df, ratings_df, n=5):
    rated_items = ratings_df[ratings_df['user_id'] == user_id]['cellphone_id'].tolist()
    unrated_items = products_df[~products_df['cellphone_id'].isin(rated_items)]['cellphone_id'].tolist()

    # Prediksi rating untuk produk yang belum dirating
    predictions = [(iid, model.predict(user_id, iid).est) for iid in unrated_items]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    result = []
    for iid, pred_rating in top_n:
        model_name = products_df[products_df['cellphone_id'] == iid]['model'].values[0]
        result.append((iid, model_name, pred_rating))

    return pd.DataFrame(result, columns=['cellphone_id', 'model', 'predicted_rating'])

# Contoh rekomendasi untuk user 1
top_recommendations_cf = get_top_n_recommendations(svd_model, user_id=1, products_df=products_df, ratings_df=ratings_df, n=5)
top_recommendations_cf

"""# **Evaluation**

# **Content Based Filterting**
"""

# Precision@K untuk Content-Based Filtering

def precision_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / k

# Simulasi evaluasi untuk user tertentu
# Misalnya kita uji untuk user_id = 1
user_id = 1

# Ambil produk yang sudah pernah dirating user (sebagai "ground truth")
relevant_items = ratings_df[ratings_df['user_id'] == user_id]['cellphone_id'].tolist()

# Ambil rekomendasi dari Content-Based untuk produk serupa (misal produk pertama)
target_product_id = products_df.iloc[0]['cellphone_id']

# Ambil Top-10 produk yang mirip (sudah dihitung di cosine_similarities sebelumnya)
idx = products_df[products_df['cellphone_id'] == target_product_id].index[0]
sim_scores = list(enumerate(cosine_similarities[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_indices = [i for i, _ in sim_scores[1:11]]  # skip itself

# Dapatkan cellphone_id hasil rekomendasi
recommended_items = products_df.iloc[sim_indices]['cellphone_id'].tolist()

# Hitung Precision@K
k = 5
precision = precision_at_k(recommended_items, relevant_items, k)
print(f"Precision@{k} untuk Content-Based Filtering (user {user_id}): {precision:.2f}")

"""# **Collaborative Filtering**"""

from surprise import accuracy

# Prediksi pada testset
predictions = svd_model.test(testset)

# Evaluasi dengan RMSE
rmse = accuracy.rmse(predictions)
print(f"RMSE Collaborative Filtering (SVD): {rmse:.4f}")

"""# **KESIMPULAN**

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
"""