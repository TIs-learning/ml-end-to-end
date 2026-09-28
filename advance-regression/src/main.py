from src.utils import set_seed
from src.data_loader import load_data
from src.preprocessing import split_data
from src.train import train_model

def main():
    # 1. Set seed
    set_seed(42)
    
    # 2. Load Data mentah
    df = load_data("data/raw/train.csv")
    
    # Catatan: Ganti 'SalePrice' jika kolom target utama dataset kamu bernama lain
    target_column = "SalePrice"
    
    # Hanya jalankan jika kolom target ada di dataset
    if target_column in df.columns:
        # Kita ambil sampel numerik saja dulu untuk testing pipeline dasar
        df_numeric = df.select_dtypes(include=['number']).fillna(0)
        
        # 3. Preprocessing / Split
        X_train, X_val, y_train, y_val = split_data(df_numeric, target_column=target_column)
        
        # 4. Train Model
        train_model(X_train, y_train, X_val, y_val, model_save_path="models/model.joblib")
    else:
        print(f"[ERROR] Kolom target '{target_column}' tidak ditemukan di dataset.")

if __name__ == "__main__":
    main()