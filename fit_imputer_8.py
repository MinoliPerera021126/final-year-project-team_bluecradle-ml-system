import numpy as np
import joblib
from sklearn.impute import SimpleImputer

data = np.load('data/prepared_data/maled_tensors.npz', allow_pickle=True)
X_train_seq = data['X_train_seq']

n_children, n_timesteps, n_features = X_train_seq.shape
seq_flat = X_train_seq.reshape(-1, n_features).astype(np.float32)
seq_flat[seq_flat == -999.0] = np.nan

print(f"Flat shape: {seq_flat.shape}")
print(f"NaN count per feature: {np.sum(np.isnan(seq_flat), axis=0)}")

imputer = SimpleImputer(strategy='mean')
imputer.fit(seq_flat)

print(f"Imputer fitted on {len(imputer.statistics_)} features.")

joblib.dump(imputer, 'models/bluecradle_imputer.pkl')
print("✅ bluecradle_imputer.pkl exported successfully.")