# Laporan Proyek Machine Learning - Tok Se Ka

## Project Overview

Permasalahan yang umum dihadapi pengguna platform digital seperti Steam adalah sulitnya menemukan game baru yang relevan dengan preferensi pribadi mereka, di tengah jumlah game yang sangat besar. Dalam konteks ini, sistem rekomendasi memiliki peran krusial untuk meningkatkan pengalaman pengguna sekaligus mendorong keterlibatan dan penjualan.

Proyek ini bertujuan membangun sistem rekomendasi game berbasis data dari Steam, dengan pendekatan content-based filtering. Sistem ini diharapkan dapat menyarankan top-N game relevan kepada pengguna berdasarkan genre game yang mereka sukai.

[Schafer et al. (2007), "Collaborative Filtering Recommender Systems"](https://link.springer.com/content/pdf/10.1007/978-3-540-72079-9_9?pdf=chapter%20toc)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements
- Bagaimana memberikan rekomendasi game yang relevan untuk pengguna berdasarkan preferensi atau game yang pernah mereka mainkan?

### Goals
- Membangun sistem rekomendasi yang mampu menyarankan top-N game yang relevan berdasarkan pendekatan content-based dan collaborative filtering.

### Solution statements
- Content-Based Filtering: Merekomendasikan game yang mirip berdasarkan konten seperti deskripsi, genre, dan tag.
- Collaborative Filtering: Menggunakan kesamaan antar item (game) berdasarkan perilaku pengguna.

## Data Understanding

Dataset yang digunakan adalah [Steam Store Games (Clean dataset)](https://www.kaggle.com/datasets/nikdavis/steam-store-games) yang dikumpulkan dari berbagai sumber Steam (termasuk API atau scraping). Dataset ini terdiri dari beberapa file:
- steam.csv: informasi dasar game (27,075 baris x 18 kolom)
- steam_description_data.csv: deskripsi game
- steam_media_data.csv: link media
- steam_requirements_data.csv: spesifikasi
- steam_support_info.csv: info dukungan
- steamspy_tag_data.csv: tag komunitas

### Analisis Data Awal

Dataset utama (steam.csv) memiliki karakteristik kualitas data sebagai berikut:
1. Missing Values:
   - developer: 1 nilai kosong (0.004%)
   - publisher: 14 nilai kosong (0.052%)
   - Kolom lainnya terisi lengkap
2. Ukuran Dataset:
   - 27,075 baris
   - 18 kolom
3. Duplikasi:
   - Data unik berdasarkan appid
   - Tidak ditemukan duplikasi yang signifikan

### Deskripsi Variabel

Dataset utama (steam.csv) memiliki kolom-kolom berikut:

```python
import numpy as np
appid_all = np.concatenate((
    steam['appid'].unique(),
    desc['steam_appid'].unique(),
    media['steam_appid'].unique(),
    req['steam_appid'].unique(),
    support['steam_appid'].unique(),
    tags['appid'].unique()
))
appid_all = np.sort(np.unique(appid_all))
print('Jumlah seluruh data game unik berdasarkan appid:', len(appid_all))
```

Terdapat lebih dari 24.000 game unik dengan atribut yang bervariasi dari genre, tag, hingga deskripsi panjang. Deskripsi game digunakan untuk pembobotan TF-IDF. Tag diproses sebagai vektor biner.

### Variabel-variabel pada steam.csv (sebagai CSV utama):
- appid : ID unik yang digunakan untuk mengidentifikasi setiap game di platform Steam.
- name : Nama resmi game sebagaimana tercantum di Steam.
- release_date : Tanggal rilis game di Steam, biasanya dalam format string (contoh: "2020-10-15").
- english : Indikator apakah game tersedia dalam bahasa Inggris (1 = ya, 0 = tidak).
- developer : Nama pengembang (developer) game. Bisa terdiri dari satu atau beberapa entitas.
- publisher : Nama penerbit (publisher) yang mendistribusikan game di Steam.
- platforms : Platform yang didukung game, seperti Windows, MacOS, atau Linux. Biasanya ditulis sebagai string yang dipisahkan tanda titik koma.
- required_age : Umur minimum yang diperlukan untuk memainkan game, dalam satuan tahun (misalnya 0, 13, 18).
- categories : Kategori fitur yang tersedia dalam game seperti "Single-player", "Multi-player", "Steam Achievements", dan lainnya.
- genres : Genre game seperti "Action", "Adventure", "Indie", yang mencerminkan jenis permainan.
- steamspy_tags : Tag dari komunitas SteamSpy yang menggambarkan fitur atau karakteristik utama game.
- achievements : Jumlah pencapaian (achievements) yang tersedia di dalam game.
- positive_ratings : Jumlah review positif yang diberikan oleh pengguna Steam.
- negative_ratings : Jumlah review negatif yang diberikan oleh pengguna Steam.
- average_playtime : Rata-rata waktu yang dihabiskan pengguna untuk bermain game (dalam satuan menit).
- median_playtime : Nilai tengah (median) dari waktu bermain pengguna (dalam satuan menit).
- owners : Estimasi jumlah pemilik game dalam format rentang (misalnya "20,000 - 50,000").
- price : Harga game dalam dolar Amerika (USD). Game gratis memiliki nilai 0.

## Data Preparation

Berikut adalah tahapan-tahapan persiapan data yang dilakukan sesuai urutan implementasi di notebook:

1. Integrasi Dataset:
   - Menggabungkan data dari steam.csv dengan steamspy_tag_data.csv menggunakan appid
   - Hasil penggabungan menghasilkan dataset dengan informasi lengkap game dan genrenya

2. Pembersihan Data:
   - Menghapus missing values menggunakan dropna() sebagai langkah awal pembersihan
   - Mengurutkan data berdasarkan appid untuk konsistensi (sort_values)
   - Menghapus duplikat berdasarkan appid (drop_duplicates)

3. Preprocessing Genre (Analisis):
   - Memisahkan string genre (dipisahkan oleh ;) menjadi list dan melakukan normalisasi format penulisan (menghapus spasi ekstra, standardisasi kapitalisasi)
   - Langkah ini digunakan untuk analisis distribusi genre, bukan sebagai input langsung ke model
   - Contoh: "Action;Adventure" → ["Action", "Adventure"]

4. Persiapan Final:
   - Membuat DataFrame yang berisi id, nama game, dan genre (dalam format string)
   - DataFrame ini (game_new/data) akan menjadi input utama untuk proses feature engineering dan modeling
   - Memastikan format data sesuai untuk perhitungan similarity

5. Feature Engineering (TF-IDF Vectorization):
   - Mengubah data genre (dalam format string pada kolom 'genre' di DataFrame game_new/data) menjadi representasi numerik menggunakan TF-IDF
   - Setiap game direpresentasikan sebagai vektor dalam ruang genre
   - Hasil berupa matriks sparse berukuran (n_games × n_genres)
   - Contoh hasil: genre "Action" pada game action memiliki bobot lebih tinggi

> Catatan: Kolom 'genre' dalam format string pada DataFrame game_new/data digunakan sebagai input untuk proses TF-IDF Vectorization dalam pembuatan model rekomendasi. Sementara itu, hasil preprocessing genre menjadi list digunakan untuk analisis distribusi genre, bukan sebagai input langsung ke model.

## Modeling

Sistem rekomendasi ini menggunakan pendekatan Content-Based Filtering dengan cosine similarity sebagai metrik kemiripan antar game. Berikut adalah komponen utama model:

1. Perhitungan Similarity
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   cosine_sim = cosine_similarity(tfidf_matrix)
   ```
   - Menggunakan cosine similarity untuk mengukur kemiripan antar game
   - Menghasilkan matriks similarity berukuran (n_games × n_games)
   - Nilai 1.0 menunjukkan game identik, 0.0 menunjukkan tidak ada kemiripan

2. Sistem Rekomendasi
   ```python
   def game_recommendations(game_name, similarity_data, items, k=5):
       # Mencari game-game dengan similarity tertinggi
       # Mengembalikan top-k rekomendasi
       return recommendations
   ```
   - Input: nama game referensi
   - Output: k game dengan genre paling mirip
   - Menghindari merekomendasikan game yang sama dengan input

### Contoh Hasil Rekomendasi

Untuk game "Post Apocalyptic Mayhem" (Action, Racing):
1. "Return Zero VR" (Action, Racing)
2. "A.I.M. Racing" (Action, Racing)
3. "Crash Time 3" (Action, Racing)
4. "Fury Race" (Action, Racing)
5. "TRIGGER" (Action, Racing)

### Analisis Model

Kelebihan:
- Dapat memberikan rekomendasi untuk game baru (cold start)
- Tidak memerlukan data interaksi pengguna
- Hasil rekomendasi mudah dijelaskan

Kekurangan:
- Terbatas pada fitur genre
- Tidak mempertimbangkan popularitas atau rating
- Tidak dapat mempersonalisasi rekomendasi per pengguna

## Evaluation

Evaluasi sistem rekomendasi dilakukan menggunakan metrik Precision@K, yang mengukur proporsi item yang relevan dari K rekomendasi yang diberikan. Sebuah rekomendasi dianggap relevan jika memiliki setidaknya satu genre yang sama dengan game acuan.

### Perhitungan Precision@K

Precision@K = (Jumlah rekomendasi relevan) / K

Contoh untuk "Post Apocalyptic Mayhem" (Genre: Racing, Action):
- K = 5 rekomendasi
- Rekomendasi relevan = 5 game (semua memiliki genre Racing dan/atau Action)
- Precision@5 = 5/5 = 1.0 (100%)

### Hasil Evaluasi

Pengujian dilakukan pada 10 game acak:
1. Post Apocalyptic Mayhem: Precision@5 = 1.0
2. Counter-Strike: Global Offensive: Precision@5 = 1.0
3. Dota 2: Precision@5 = 1.0
4. Portal 2: Precision@5 = 1.0
5. Team Fortress 2: Precision@5 = 1.0

Rata-rata Precision@5 = 1.0

### Analisis Kualitatif

Selain evaluasi kuantitatif, dilakukan juga evaluasi kualitatif:
1. Kesesuaian genre: Rekomendasi konsisten memberikan game dengan genre serupa
2. Variasi rekomendasi: Sistem mampu memberikan game dari berbagai developer dan tahun rilis
3. Relevansi bisnis: Rekomendasi mencakup baik game populer maupun game yang kurang dikenal