import sys
sys.path.insert(0, '.')
from app import load_data, get_model_config
data = load_data()
cfg = get_model_config(data)
print(f'Data loaded: windows={data["windows"].shape}')
print(f'Config: n_stocks={cfg["n_stocks"]}, T={cfg["T"]}, in_feats={cfg["in_feats"]}')
print('SUCCESS: No dimension mismatch')