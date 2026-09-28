import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train, X_val, y_val, model_save_path: str = "models/model.joblib"):
    """Melatih model RandomForest, menghitung evaluasi RMSE, dan menyimpan model."""
    print("[INFO] Melatih model RandomForest...")
    
    # Contoh model regresi sederhana
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    print(f"[INFO] Validation RMSE: {rmse:.4f}")
    
    # Simpan model yang sudah dilatih
    joblib.dump(model, model_save_path)
    print(f"[INFO] Model berhasil disimpan ke '{model_save_path}'")
    
    return model