# %% [markdown]
# ## Problem Definition
# 
# ---
# 
# ## 1. Background
# Dalam bidang real estate, penentuan harga jual property (property valuation) merupakan langkah yang sangat krusial. Nilai jual yang dimiliki dari sebuah property ditentukan berdasarkan kombinasi puluhan faktor (fitur), mulai dari luas tanah, kualitas material, usia bangunan, hingga kondisi lingkungan disekitar.
# 
# Namun, penentuan harga property secara manual berdasarkan intuisi tanpa analisis data yang matang memiliki risiko yang tinggi, sehingga dapat mengakibatkan:
# 
# - **Underpricing**: Harga terlalu murah yang akan membuat penjual / agen property rugi margin (kurang optimal). 
#     
# - **Overpricing**: Harga property terlalu tinggi sehingga mengakibatkan property susah untuk laku dan tidak kompetitif.
# 
# Dengan terdapatnya puluhan fitur yang saling berinteraksi, penilaian manual menjadi lambat dan subjektif. Oleh karena itu, dibutuhkan pendekatan Machine Learning untuk menentukan harga jual yang sesuai berdasarkan data yang dimiliki property yang ingin dijual.
# 
# ---
# 
# ## 2. Business Objective
# - Memprediksi harga jual rumah ('SalePrice') secara objektif berdasarkan feature yang dimiliki oleh property.
# 
# - Meningkatkan efisiensi waktu penilai atau agen property untuk melakukan **property valuation** serta meminimalkan kesalahan dalam penetapan harga jual.
# 
# - Mengidentifikasi fitur penentu property valuation yang paling dominan (feature importance) dalam meningkatkan nilai property.
# 
# ---
# 
# ## 3. Problem Definition
# 
# ### Business Question
# > Berapa estimasi ('Sale Price') berdasarkan Kondisi Bangunan, Kondisi sekitar, Tahun Pembangunan, Arsitektur setiap bagian rumah?
# 
# ### Machine Learning Framing
# - **Problem Type**: Supervised Learning
# - **Task**: Regression (Continuous Value Prediction).
# - **Target Variable**: `Sale Price` (Harga jual property).
# - **Evaluation Metric**: RMSLE (Root Mean Squared Logarithmic Error).
# 
# ---
# 
# ## 4. Input Data
# 
# Dataset berisi data fiture property dengan kolom sebagai berikut:
# 
# ### Numerical Features
# 
# **Area & Ukuran (Square Feet / sq ft)**
# - **`LotFrontage`**
# Panjang garis depan tanah yang berbatasan langsung dengan jalan (kaki)
# - **`LotArea`**
# Total luas area tanah (sq ft).
# - **`MasVnrArea`**
# Luas area lapisan dinding batu eksterior (masonry veneer).
# - **`BsmtFinSF1`**
# Luas basement area 1 yang sudah di-finish.
# - **`BsmtFinSF2`**
# Luas basement area 2 yang sudah di-finish.
# - **`BsmtUnfSF`**
# Luas area basement yang belum di-finish.
# - **`TotalBsmtSF`**
# Total luas keseluruhan basement.
# - **`1stFlrSF`**
# Luas lantai 1 (sq ft).
# - **`2ndFlrSF`**
# Luas lantai 2 (sq ft).
# - **`LowQualFinSF`**
# Luas area dengan finishing kualitas rendah di semua lantai.
# - **`GrLivArea`**
# Total luas area tinggal di atas permukaan tanah (Ground Living Area).
# - **`GarageArea`**
# Luas area garasi (sq ft).
# - **`WoodDeckSF`**
# Luas area dek kayu di luar rumah.
# - **`OpenPorchSF`**
# Luas teras terbuka.
# - **`EnclosedPorch`**
# Luas teras tertutup.
# - **`3SsnPorch`**
# Luas teras tiga musim (three season porch).
# - **`ScreenPorch`**
# Luas teras berbahan kawat/kaca pelindung.
# - **`PoolArea`**
# Luas area kolam renang.
# 
# **Jumlah Kamar & Fasilitas**
# - **`BsmtFullBath`**
# Jumlah kamar mandi utama (penuh) di basement.
# - **`BsmtHalfBath`**
# Jumlah kamar mandi setengah (tanpa shower) di basement.
# - **`FullBath`**
# Jumlah kamar mandi utama (penuh) di atas tanah.
# - **`HalfBath`**
# Jumlah kamar mandi setengah di atas tanah.
# - **`BedroomAbvGr`**
# Jumlah kamar tidur di atas tanah.
# - **`KitchenAbvGr`**
# Jumlah dapur di atas tanah.
# - **`TotRmsAbvGrd`**
# Total seluruh ruangan di atas tanah (tidak termasuk kamar mandi).
# - **`Fireplaces`**
# Jumlah perapian di dalam rumah.
# - **`GarageCars`**
# Kapasitas garasi (jumlah mobil yang bisa masuk).
# 
# **Tahun & Waktu**
# - **`YearBuilt`**
# Tahun rumah pertama kali dibangun.
# - **`YearRemodAdd`**
# Tahun rumah direnovasi (sama dengan YearBuilt jika belum pernah renovasi).
# - **`GarageYrBlt`**
# Tahun garasi dibangun.
# - **`MoSold`**
# Bulan transaksi penjualan (1-12).
# - **`YrSold`**
# Tahun transaksi penjualan.
# 
# **Skala Penilaian Kuantitatif**
# - **`MSSubClass`**
# Kode angka jenis properti/bangunan (misal: 20 = 1-Story 1946 & newer).
# - **`OverallQual`**
# Rating kualitas material dan finishing keseluruhan rumah (skala 1–10).
# - **`OverallCond`**
# Rating kondisi fisik rumah saat ini (skala 1–10).
# - **`MiscVal`**
# Nilai mata uang dari fitur tambahan yang ada di MiscFeature (USD).
# 
# ---
# 
# ### Categorical Features
# 
# **Lokasi & Karakteristik Tanah**
# - **`MSZoning`**
# Klasifikasi zonasi wilayah (misal: Residensial padat, Residensial longgar, Komersial).
# - **`Street`**
# Jenis akses jalan ke properti (Gravel/Paved).
# - **`Alley`**
# Jenis akses gang ke properti (Grvl/Pave/No Alley).
# - **`LotShape`**
# Bentuk fisik tanah (Regular, Slightly irregular, Irregular).
# - **`LandContour`**
# Kerataan tanah (Flat, Hillside, Depressed).
# - **`Utilities`**
# Ketersediaan fasilitas publik (AllPub, NoSewer, NoSeWa).
# - **`LotConfig`**
# Konfigurasi lot tanah (Inside lot, Corner lot, Cul-de-sac).
# - **`LandSlope`**
# Kemiringan tanah properti (Gentle, Moderate, Severe).
# - **`Neighborhood`**
# Nama lokasi lingkungan/lokasi perumahan di kota Ames.
# - **`Condition1 & Condition2`**
# Kedekatan dengan kondisi khusus (misal: dekat jalan tol, rel kereta api).
# 
# **Bentuk & Material Eksterior**
# - **`BldgType`**
# Tipe hunian (Single-family Detached, Townhouse, Duplex).
# - **`HouseStyle`**
# Gaya arsitektur bangunan (1 Story, 2 Story, Split Foyer).
# - **`RoofStyle`**
# Jenis/bentuk atap (Gable, Hip, Mansard).
# - **`RoofMatl`**
# Bahan baku atap (Standard Shingle, Wood Shake, Metal).
# - **`Exterior1st`**
# Pelapis dinding luar rumah (bagian utama).
# - **`Exterior2nd`**
# Pelapis dinding luar rumah (bagian sekunder jika ada).
# - **`MasVnrType`**
# Tipe bahan lapisan batu dinding (Masonry veneer).
# - **`ExterQual`**
# Kualitas material eksterior (Ex, Gd, TA, Fa, Po).
# - **`ExterCond`**
# Kondisi fisik material eksterior saat ini.
# 
# **Pondasi & Basement**
# - **`Foundation`**
# Jenis bahan pondasi (Poured Concrete, Cinder Block, Wood).
# - **`BsmtQual`**
# Ketinggian dan kualitas ruang basement.
# - **`BsmtCond`**
# Kondisi umum ruang basement.
# - **`BsmtExposure`**
# Tingkat akses dinding basement ke ruang terbuka/cahaya luar (Walkout, Exposure).
# - **`BsmtFinType1`**
# Kualitas finishing area basement utama.
# - **`BsmtFinType2`**
# Kualitas finishing area basement sekunder.
# 
# **Sistem Dalam Rumah & Fasilitas**
# - **`Heating`**
# Jenis sistem pemanas ruangan.
# - **`HeatingQC`**
# Kualitas dan kondisi sistem pemanas.
# - **`CentralAir`**
# Adanya sistem pendingin udara pusat/AC (Y/N).
# - **`Electrical`**
# Jenis sistem kelistrikan rumah.
# - **`KitchenQual`**
# Kualitas area dapur.
# - **`Functional`**
# Tingkat fungsi utilitas rumah secara keseluruhan (Home functionality).
# - **`FireplaceQu`**
# Kualitas perapian.
# - **`PavedDrive`**
# Jenis halaman jalan masuk mobil (paved driveway).
# 
# **Garasi, Kolam, & Fitur Tambahan**
# - **`GarageType`**
# Lokasi dan tipe garasi (Attached, Detached, Built-in).
# - **`GarageFinish`**
# Tingkat finishing interior garasi (Fin, RFn, Unf).
# - **`GarageQual`**
# Kualitas fisik garasi.
# - **`GarageCond`**
# Kondisi fisik garasi.
# - **`PoolQC`**
# Kualitas kolam renang.
# - **`Fence`**
# Kualitas dan tipe pagar rumah.
# - **`MiscFeature`**
# Fitur khusus tambahan yang tidak terliput kolom lain (Elevator, Shed, Tennis Court).
# 
# **Jenis Transaksi**
# - **`SaleType`**
# Jenis dokumen/skema penjualan (Warranty Deed, Cash, Court Officer).
# - **`SaleCondition`**
# Kondisi khusus transaksi (Normal sale, Allocation, Foreclosure/Lelang).
# 
# ---
# 
# ### Identifier Features
# - **`Id`**
# Nomor unik identitas setiap transaksi rumah.
# 
# ---
# 
# ### Target Variable
# - **`SalePrice`**
# Harga jual rumah dalam mata uang USD (variabel yang akan diprediksi).
# 
# ---
# 
# ## 5. Data Notes & Constraints
# Keterangan Data akan di isi setelah selesai melakukan eda
# 
# ---
# 
# ## 6. Expected Outcome
# - Model Teknis: Model machine learning regresi dengan performa $RMSLE$ yang rendah pada data test.
# 
# - Wawasan Bisnis (Business Insight): Mengidentifikasi fitur mana saja yang paling memengaruhi harga (Feature Importance).
# 
# - Pipeline Data: Sistem/fungsi kode yang rapi dari Data Loading, Preprocessing, hingga Inference (prediksi data baru).

# %% [markdown]
# ## Data Acquistion
# 
# ---

# %%
# import library
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("../data/raw/train.csv")

# %% [markdown]
# ## Data Cleaning & Understanding

# %%
# Drop ident feature
df = df.drop(columns=['Id'])

# %%
# Keterangan setiap columns
df.info()


