import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Memisahkan fitur (X) dan target (y) serta membagi dataset ke Train & Validation set."""
    if target_column not in df.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di DataFrame.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"[INFO] Data split berhasil! Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val