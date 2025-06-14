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

### Split dataset
Dalam membangun model pembelajaran mesin, dataset perlu dibagi menjadi dua bagian utama:

- Data Latih: Digunakan untuk melatih model. Model akan mempelajari pola dan hubungan antara fitur dan target dalam data latih ini.
- Data Uji: Digunakan untuk menguji kinerja model yang telah dilatih. Model akan memprediksi nilai pada data uji, dan hasilnya akan dibandingkan dengan nilai sebenarnya untuk mengevaluasi seberapa baik model tersebut bekerja pada data yang belum pernah dilihat sebelumnya.
Pada proyek ini penulis membagi dataset menjadi 80:20. Pembagian data dengan perbandingan 80:20 adalah salah satu cara yang umum digunakan, artinya 80% data akan digunakan untuk melatih model dan 20% sisanya untuk menguji model.

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

### Persiapan untuk Content-Based Filtering
Menghitung kemiripan antar produk ponsel berdasarkan fitur teks yang digabung dari kolom brand, model, dan RAM. Pertama, dibuat kolom features yang menyatukan informasi tersebut menjadi deskripsi singkat setiap produk, misalnya "Samsung Galaxy A12 4GB RAM". Kemudian, deskripsi ini dikonversi menjadi representasi numerik menggunakan teknik TF-IDF (TfidfVectorizer) dengan menghapus stop words bahasa Inggris agar fokus pada kata-kata penting. Hasil vektorisasi ini disimpan dalam tfidf_matrix, yang selanjutnya digunakan untuk menghitung cosine similarity antar produk, sehingga menghasilkan matriks cosine_similarities yang menunjukkan seberapa mirip setiap produk dengan produk lainnya berdasarkan konten teksnya. 

Dengan melakukan ekstraksi fitur dari teks menggunakan TF-IDF (Term Frequency–Inverse Document Frequency), sebuah teknik untuk mengubah data teks menjadi representasi numerik berbobot berdasarkan frekuensi kemunculan kata yang dianggap penting. Dengan TfidfVectorizer(), data teks dari kolom features diubah menjadi matriks TF-IDF (tfidf_matrix), di mana setiap baris mewakili satu produk dan setiap kolom mewakili satu kata unik yang muncul di seluruh deskripsi produk. Nilai dalam matriks mencerminkan pentingnya kata tersebut dalam deskripsi produk tertentu dibandingkan dengan semua produk lainnya.

Kemudian, menghitung kemiripan antar produk berdasarkan fitur teks yang telah diekstraksi menggunakan TF-IDF. Dengan menggunakan cosine_similarity, setiap pasangan produk dibandingkan berdasarkan sudut (cosine) antara vektor representasi mereka dalam ruang fitur. Hasilnya disimpan dalam matriks cosine_sim, di mana setiap elemen [i][j] menunjukkan tingkat kemiripan antara produk ke-i dan ke-j. Nilai cosine similarity berkisar antara 0 hingga 1, di mana nilai yang lebih tinggi menunjukkan tingkat kemiripan yang lebih besar.

### Persiapan untuk Collaborative Filtering (SVD)
Pertama, memilih kolom-kolom penting dari DataFrame ratings_df, yaitu user_id, cellphone_id, dan rating. Ketiga kolom ini merupakan informasi utama yang diperlukan dalam sistem rekomendasi berbasis rating, di mana user_id mengidentifikasi pengguna, cellphone_id mengidentifikasi produk yang dinilai, dan rating menunjukkan nilai penilaian yang diberikan pengguna terhadap produk tersebut. Dengan menyederhanakan DataFrame hanya pada kolom-kolom ini, proses analisis dan pemodelan dapat dilakukan dengan lebih fokus dan efisien.

Kemudian, memformat data rating agar sesuai dengan struktur yang dibutuhkan oleh library Surprise, yang digunakan untuk membangun sistem rekomendasi. Objek Reader digunakan untuk menentukan skala rating, dalam hal ini dari 1 hingga 10. Kemudian, Dataset.load_from_df() digunakan untuk mengubah DataFrame ratings_df menjadi format internal Surprise, di mana hanya tiga kolom yang digunakan secara berurutan: user_id, cellphone_id, dan rating. Format ini memungkinkan data untuk diproses lebih lanjut oleh algoritma rekomendasi dalam library Surprise.

Setelah itu,  membagi data rating ke dalam dua bagian, yaitu data pelatihan (trainset) dan data pengujian (testset), dengan menggunakan fungsi train_test_split dari library Surprise. Proporsi data pengujian ditetapkan sebesar 20% (test_size=0.2), sedangkan sisanya digunakan untuk melatih model rekomendasi. Parameter random_state=42 digunakan agar pembagian data bersifat konsisten setiap kali kode dijalankan. Pembagian ini penting untuk mengevaluasi kinerja model secara objektif dengan membandingkan prediksi terhadap data yang belum pernah dilihat selama pelatihan.

---

## Modeling
Model rekomendasi dibangun dengan dua pendekatan utama:
### **Content Based Filtering**
Model Content-Based Filtering (CBF) pada proyek ini dibangun dengan pendekatan berbasis kemiripan fitur dari masing-masing produk ponsel. Untuk merepresentasikan fitur produk dalam bentuk numerik, digunakan teknik TF-IDF (Term Frequency–Inverse Document Frequency) terhadap kolom 'features' yang berisi gabungan informasi dari brand, model, dan kapasitas RAM. Proses ini menghasilkan vektor TF-IDF untuk setiap produk, yang kemudian digunakan untuk menghitung cosine similarity antar produk. Nilai cosine similarity menunjukkan tingkat kemiripan antara dua produk, dengan rentang antara 0 (tidak mirip) hingga 1 (sangat mirip).

Selanjutnya, fungsi recommend_content_based() dibuat untuk menghasilkan rekomendasi produk yang mirip dengan suatu produk input berdasarkan nilai kemiripan tersebut. Fungsi ini bekerja dengan menemukan indeks produk berdasarkan cellphone_id, lalu memilih produk-produk dengan skor kemiripan tertinggi (selain produk itu sendiri). Hasil akhirnya berupa daftar produk yang direkomendasikan beserta skor kemiripannya. Dengan pendekatan ini, sistem mampu merekomendasikan produk yang memiliki karakteristik serupa, meskipun belum pernah dinilai oleh pengguna yang sama.

![image](https://github.com/user-attachments/assets/705b9a85-faca-404e-9d2f-e1d3461b8577)


### **Collaborative Filtering**
Model Collaborative Filtering (CF) pada sistem rekomendasi ini menggunakan pendekatan Matrix Factorization dengan algoritma Singular Value Decomposition (SVD) yang disediakan oleh library Surprise. Data rating dari pengguna diformat ulang menggunakan objek Reader, dengan skala rating antara 1 hingga 10. Setelah itu, data dibagi menjadi data pelatihan dan pengujian dengan proporsi 80:20 untuk memastikan evaluasi model dilakukan secara adil terhadap data yang belum pernah dilihat.

Model SVD kemudian dilatih menggunakan data pelatihan, di mana algoritma ini mempelajari pola preferensi pengguna terhadap produk dengan cara memetakan pengguna dan item ke dalam dimensi laten. Untuk menghasilkan rekomendasi, dibuat fungsi get_top_n_recommendations() yang memprediksi rating produk-produk yang belum pernah diberi penilaian oleh pengguna tertentu, lalu memilih N produk dengan prediksi rating tertinggi. Hasilnya berupa daftar rekomendasi yang dipersonalisasi untuk pengguna berdasarkan interaksi pengguna lain yang memiliki pola preferensi serupa. Pendekatan ini efektif dalam menangkap pola kolektif dari perilaku pengguna meskipun tidak mempertimbangkan informasi konten produk secara eksplisit.

![image](https://github.com/user-attachments/assets/9b8450b9-7c89-4f85-967f-61190678d8c8)

##  Evaluation
### **Content-Based Filtering**

Evaluasi performa model Content-Based Filtering dilakukan dengan menggunakan metrik Precision@K, yaitu rasio jumlah item yang relevan di antara K item teratas yang direkomendasikan. Dalam simulasi ini, digunakan data dari pengguna dengan user_id = 1, di mana daftar produk yang pernah diberi rating oleh pengguna tersebut dianggap sebagai ground truth. Kemudian, sistem merekomendasikan 10 produk yang paling mirip dengan salah satu produk yang pernah dilihat oleh pengguna (berdasarkan cosine similarity dari fitur produk), dan Precision@5 dihitung berdasarkan seberapa banyak dari 5 produk teratas tersebut yang memang pernah diberi rating oleh pengguna.

Hasil evaluasi menunjukkan bahwa Precision@5 untuk model CBF adalah 0.40, yang berarti 40% dari rekomendasi teratas berhasil menampilkan produk yang relevan bagi pengguna. Meskipun hasil ini belum tentu optimal, nilai tersebut memberikan indikasi bahwa model mampu menangkap sebagian preferensi pengguna melalui kemiripan konten antar produk. Namun demikian, untuk meningkatkan akurasi rekomendasi, pendekatan ini dapat digabungkan dengan model lain seperti Collaborative Filtering atau Hybrid Recommender.

![image](https://github.com/user-attachments/assets/f6725aa9-966b-419a-929a-65a14dc451c4)

---

### **Collaborative Filtering**
Evaluasi model Collaborative Filtering yang dibangun dengan algoritma Singular Value Decomposition (SVD) dilakukan menggunakan metrik Root Mean Squared Error (RMSE). RMSE mengukur selisih rata-rata kuadrat antara rating yang diprediksi oleh model dan rating aktual dari pengguna, kemudian diakarkan untuk menjaga satuan tetap konsisten dengan skala rating. Semakin kecil nilai RMSE, semakin baik performa model dalam memprediksi rating.

Dalam evaluasi ini, model diuji pada data pengujian (testset) yang sebelumnya tidak digunakan selama proses pelatihan. Hasil evaluasi menunjukkan bahwa model menghasilkan RMSE sebesar 2.2279, yang menunjukkan bahwa prediksi model masih memiliki selisih rata-rata sekitar 2.2 poin dari nilai rating aktual pada skala 1 sampai 10. Nilai ini masih cukup wajar dalam sistem rekomendasi umum, namun dapat ditingkatkan lebih lanjut dengan fine-tuning parameter atau menggunakan pendekatan hybrid.

![image](https://github.com/user-attachments/assets/424818c6-82b0-4cca-b656-bdfcabbd99da)


# **Evaluasi Terhadap Business Understanding**
## ✅ Problem Statements
1.  Bagaimana cara membantu pengguna menemukan smartphone yang sesuai dengan kebutuhan mereka di tengah banyaknya pilihan yang tersedia?
  `Melalui pendekatan Content-Based Filtering, sistem mampu merekomendasikan produk yang memiliki spesifikasi teknis serupa, seperti brand, model, RAM, kamera, baterai, dan harga. Hal ini sangat membantu pengguna yang memiliki preferensi fitur tertentu dalam menyaring pilihan yang relevan.`
  
2. Bagaimana sistem dapat memberikan rekomendasi yang relevan agar pengguna tidak kebingungan dalam mengambil keputusan pembelian?
  `Dengan pendekatan Collaborative Filtering berbasis algoritma SVD, sistem memberikan rekomendasi personal berdasarkan pola rating pengguna lain yang memiliki preferensi serupa. Ini membantu pengguna dalam mengambil keputusan yang lebih tepat tanpa harus mencoba semua produk secara manual.`

## ✅ Goals
1. Membangun sistem rekomendasi smartphone berbasis konten dan kolaborasi.
   `Telah tercapai. Dua pendekatan utama—Content-Based Filtering dan Collaborative Filtering (SVD)—telah berhasil diimplementasikan dan diuji, masing-masing berfokus pada kemiripan fitur produk dan pola preferensi pengguna.`
   
2. Memberikan rekomendasi yang akurat untuk meningkatkan kepuasan pengguna.
   `Sebagian besar tercapai. Evaluasi menunjukkan bahwa Content-Based Filtering mampu memberikan produk alternatif yang mirip, sementara Collaborative Filtering menghasilkan RMSE sebesar 2.2279, yang cukup baik pada skala rating 1–10.`

## ✅ Solution Approach
1. **Content-Based Filtering**: Menggunakan atribut produk (brand, model, RAM, kamera, baterai, dan harga) untuk mengukur kesamaan antar produk.
   `Menggunakan teknik TF-IDF dan cosine similarity untuk mengukur kesamaan antar produk berdasarkan atribut seperti brand, model, RAM, kamera, baterai, dan harga. Rekomendasi dihasilkan dengan menampilkan produk-produk yang memiliki skor kemiripan tinggi terhadap produk target.`
  
2. **Collaborative Filtering (SVD)**: Menggunakan data rating dari pengguna terhadap produk untuk merekomendasikan smartphone yang disukai oleh pengguna dengan preferensi serupa.
   `Menggunakan data interaksi pengguna (rating) terhadap produk, model SVD membangun matriks faktor tersembunyi yang digunakan untuk memprediksi rating produk yang belum pernah diulas oleh pengguna. Rekomendasi disesuaikan secara personal berdasarkan perilaku pengguna lain yang serupa.`
   
✅ Berhasil diterapkan. Evaluasi menggunakan RMSE menunjukkan nilai sebesar 2.2279, yang mencerminkan prediksi yang cukup baik terhadap preferensi pengguna.
   
---


# KESIMPULAN
## 1. Content-Based Filtering

- Sistem membandingkan produk berdasarkan kemiripan atribut teknis, seperti RAM, kamera, baterai, dan harga.
- Menerapkan TF-IDF Vectorizer dan cosine similarity untuk menghitung tingkat kesamaan antar produk.
- Evaluasi menggunakan Precision@5 menunjukkan nilai 0.40, yang berarti 40% dari rekomendasi berada dalam daftar produk yang telah dirating oleh user, menandakan relevansi yang cukup baik.

📌 **Kelebihan**:
- Efektif untuk produk baru yang belum memiliki rating.
- Menyediakan alternatif dari brand berbeda berdasarkan spesifikasi.

---

## 2. Collaborative Filtering (SVD)

- Model belajar dari perilaku pengguna untuk memprediksi rating produk yang belum pernah mereka coba.
- Menggunakan algoritma SVD dari pustaka Surprise untuk membuat prediksi berbasis matriks interaksi.
- Hasil evaluasi menunjukkan nilai RMSE sebesar 2.2279, yang termasuk baik untuk skala rating 1–10.

📌 **Kelebihan**:
- Memberikan rekomendasi yang sangat personal dan relevan.
- Cocok untuk pengguna aktif yang sudah memberikan banyak rating.

---


## Kesimpulan Sistem

Secara keseluruhan, sistem rekomendasi yang dibangun berhasil memenuhi tujuan utama, yaitu memberikan saran produk yang relevan dan membantu pengguna dalam proses pengambilan keputusan. Pendekatan Content-Based Filtering efektif dalam merekomendasikan produk serupa berdasarkan spesifikasi teknis, sedangkan Collaborative Filtering memberikan saran personal yang lebih sesuai dengan preferensi pengguna. Dengan kombinasi kedua pendekatan ini, sistem memiliki potensi untuk dikembangkan lebih lanjut menjadi hybrid recommender system guna memaksimalkan akurasi dan cakupan rekomendasi.

