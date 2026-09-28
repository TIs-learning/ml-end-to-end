import os
import random
import numpy as np

def set_seed(seed: int = 42):
    """Mengatur random seed agar hasil eksperimen dapat direproduksi."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Random seed diset ke: {seed}")