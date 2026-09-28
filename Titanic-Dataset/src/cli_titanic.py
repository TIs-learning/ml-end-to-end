import pandas as pd
import numpy as np
import joblib


# 1. Load semua objek yang sudah dilatih dari folder models
# (Pastikan path foldernya benar)
scaler = joblib.load('../models/scaler.joblib')
preprocess = joblib.load('../models/encoder.joblib')
best_knn = joblib.load('../models/best_knn_model.joblib')
X_train_columns = joblib.load('../models/x_train_columns.joblib') # Jika kamu butuh urutan kolom aslinya

# 2. Barulah tempel fungsi CLI-mu di bawah sini
def terminal_titanic():
    print("=============================================")
    print("   SISTEM PREDIKSI KESELAMATAN TITANIC       ")
    print("=============================================")
    print("Masukkan data penumpang baru:")
    
    try:
        # 1. Mengambil Input Lengkap
        pclass = int(input("Kelas Tiket (1, 2, atau 3)       : "))
        sex = input("Jenis Kelamin (male/female)        : ").strip().lower()
        age = float(input("Umur (angka)                       : "))
        sibsp = int(input("Jumlah Saudara/Pasangan            : "))
        parch = int(input("Jumlah Orang Tua/Anak              : "))
        fare = float(input("Harga Total Tiket ($)              : "))
        embarked = input("Pelabuhan (S/C/Q)                  : ").strip().upper()
        title = input("Gelar (Mr/Mrs/Miss/Master/Rare)    : ").strip()
        deck = input("Dek Kabin (B/C/D/E/F/G/T/U)        : ").strip().upper()
        
        print("\n[INFO] Memproses data ke dalam mesin KNN...")
        
        # 2. Menerapkan Logika Feature Engineering
        family_size = sibsp + parch
        fare_per_person = fare / (family_size + 1)
        
        # Penamaan harus presisi (Familysize dengan s kecil)
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Age': [age],
            'Sex': [sex],
            'Embarked': [embarked],
            'Title': [title],
            'Deck': [deck],
            'Familysize': [family_size],
            'FarePerPerson': [fare_per_person]
        })
        
        # 3. Proses One-Hot Encoding
        cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
        encoded_features = ohe.transform(input_data[cat_cols])
        encoded_df = pd.DataFrame(
            encoded_features, 
            columns=ohe.get_feature_names_out(cat_cols),
            index=input_data.index
        )
        
        # Gabungkan dan buang teks asli
        input_final = pd.concat([input_data.drop(columns=cat_cols), encoded_df], axis=1)
        
        # 4. KUNCI REKAYASA: Sinkronisasi 19 Kolom X_train
        input_final = input_final.reindex(columns=X_train.columns, fill_value=0)
        
        # 5. SURGICAL SCALING: Hanya menskalakan 3 kolom kontinyu!
        cont_cols = ['Age', 'Familysize', 'FarePerPerson']
        input_final[cont_cols] = scaler.transform(input_final[cont_cols])
        
        # 6. Prediksi Akhir dengan Model Terbaik
        prediksi = best_knn.predict(input_final)
        
        print("=============================================")
        if prediksi[0] == 1:
            print(">> HASIL: PENUMPANG DIPREDIKSI [SELAMAT] (SURVIVED) <<")
        else:
            print(">> HASIL: PENUMPANG DIPREDIKSI [TENGGELAM] (NOT SURVIVED) <<")
        print("=============================================")
            
    except Exception as e:
        print(f"\n[ERROR] Komputasi gagal. Detail: {e}")
        
if __name__ == "__main__":
    terminal_titanic()