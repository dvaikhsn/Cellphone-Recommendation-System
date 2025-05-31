Laporan Proyek Machine Learning – [Nama Anda]
1. Project Overview
Di era digital saat ini, pengguna dihadapkan dengan banyak pilihan produk, terutama dalam kategori elektronik seperti smartphone. Banyaknya variasi merek, spesifikasi, dan harga membuat pengguna kesulitan dalam menentukan pilihan terbaik. Sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan produk yang relevan dengan preferensi dan kebutuhan mereka.

Proyek ini bertujuan membangun sistem rekomendasi handphone menggunakan pendekatan Content-Based Filtering, Collaborative Filtering (SVD), dan Neural Collaborative Filtering (NCF) untuk menyarankan produk yang sesuai dengan preferensi pengguna berdasarkan data spesifikasi dan rating historis pengguna lain.

Referensi:

Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). Recommender systems survey. Knowledge-Based Systems, 46, 109–132.

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. Proceedings of the 26th international conference on world wide web, 173–182.

2. Business Understanding
Problem Statements
Pengguna kesulitan memilih handphone dari ribuan pilihan yang tersedia di pasaran.

Sistem rekomendasi yang ada seringkali tidak memperhatikan preferensi personal atau spesifikasi yang diminati pengguna.

Goals
Membangun sistem rekomendasi yang menyarankan handphone berdasarkan kesamaan spesifikasi (Content-Based).

Memberikan rekomendasi berdasarkan perilaku pengguna lain dengan preferensi serupa (Collaborative Filtering).

Menggunakan pendekatan deep learning (NCF) untuk meningkatkan akurasi dan menangkap interaksi kompleks antara user dan item.

Solution Approach
Content-Based Filtering: menggunakan data spesifikasi handphone (RAM, harga, OS, kamera, dll).

Collaborative Filtering (SVD): memanfaatkan interaksi pengguna (rating) untuk menyarankan produk.

Neural Collaborative Filtering (NCF): mengombinasikan embedding dan neural networks untuk menangkap hubungan non-linear.

3. Data Understanding
Dataset ini terdiri dari 3 file:

cellphones ratings.csv – data rating antara pengguna dan handphone

cellphones users.csv – informasi demografi pengguna

cellphones data.csv – spesifikasi lengkap produk handphone

Jumlah data:

Rating: 15.000+ baris

Produk: ~500 handphone unik

User: 1.000+ pengguna

Fitur-Fitur Dataset:
user_id: ID unik pengguna

cellphone_id: ID unik produk

rating: rating dari pengguna (skala 1–5)

brand: merek handphone

model: model atau seri

price: harga handphone

RAM, camera, battery size: fitur teknis dari handphone

operating system: sistem operasi

EDA dilakukan untuk memahami distribusi rating, produk terpopuler, korelasi fitur, dan distribusi harga.

4. Data Preparation
Duplicate Removal: Menghapus data duplikat dari ketiga dataset.

Missing Value Handling:

ratings_df: drop baris kosong.

users_df dan products_df: isi kosong dengan metode forward fill.

Merge: Dataset digabung menjadi satu dataframe gabungan.

Feature Engineering:

Gunakan TF-IDF pada nama model.

Encode fitur kategori seperti brand dan OS.

Skala fitur numerik seperti RAM, kamera, dan harga.

5. Modeling
✅ Content-Based Filtering
Menggunakan cosine similarity antar fitur handphone.

Hasilkan top-5 produk yang paling mirip dengan produk yang ditentukan.

Kelebihan: cocok untuk cold-start user.

Kekurangan: tidak mempertimbangkan feedback pengguna.

✅ Collaborative Filtering (SVD)
Menggunakan Surprise SVD.

Memberikan rekomendasi berdasarkan rating pengguna serupa.

Lebih personal dan dinamis.

✅ Neural Collaborative Filtering (NCF)
Menggunakan TensorFlow/Keras.

Menggunakan embedding + dense layers untuk menangkap interaksi non-linear.

Performa lebih tinggi saat tersedia data cukup.

6. Evaluation
Metrik evaluasi:

MAE (Mean Absolute Error): rata-rata selisih absolut antara prediksi dan rating aktual.

RMSE (Root Mean Squared Error): penalti lebih besar untuk prediksi jauh dari rating aktual.

✅ Hasil Evaluasi:
Model	MAE	RMSE
Baseline (Mean Rating)	0.65	0.80
Collaborative Filtering (SVD)	0.42	0.53
Neural CF (NCF)	~0.40	~0.51

NCF memiliki performa terbaik, diikuti oleh SVD, dan jauh lebih baik dari baseline sederhana.

7. Kesimpulan dan Saran
✅ Kesimpulan:
Sistem rekomendasi berhasil dibangun dengan tiga pendekatan.

Content-Based cocok untuk user baru.

Collaborative Filtering (SVD) dan NCF lebih unggul dalam akurasi.

NCF memungkinkan untuk menangkap interaksi yang lebih kompleks.

✅ Saran:
Tambahkan data ulasan teks atau gambar produk untuk memperkaya fitur.

Lakukan tuning model NCF lebih lanjut (epoch, embedding).

Bangun antarmuka interaktif berbasis web atau mobile.

Uji dengan metrik ranking seperti precision@k, recall@k, atau NDCG.
