import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Membaca file CSV dataset."""
    df = pd.read_csv(file_path)
    print(f"[INFO] Data berhasil dimuat dari '{file_path}' (Shape: {df.shape})")
    return df