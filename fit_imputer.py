import numpy as np
import joblib
from sklearn.impute import SimpleImputer

# Load MAL-ED tensors
data = np.load('data/prepared_data/maled_tensors.npz', allow_pickle=True)
X_train_seq = data['X_train_seq']  # shape: (560, 6, 8)

print(f"X_train_seq shape: {X_train_seq.shape}")

# Reshape from (560, 6, 8) to (3360, 8) — one row per visit
n_children, n_timesteps, n_features = X_train_seq.shape
seq_flat = X_train_seq.reshape(-1, n_features)

print(f"Flattened shape: {seq_flat.shape}")

# Replace padding value -999.0 with NaN before fitting
seq_flat = seq_flat.astype(np.float32)
seq_flat[seq_flat == -999.0] = np.nan

print(f"NaN count per feature: {np.sum(np.isnan(seq_flat), axis=0)}")

# Fit SimpleImputer with mean strategy on the 8 sequence features
imputer = SimpleImputer(strategy='mean')
imputer.fit(seq_flat)

print(f"Imputer fitted. Feature means: {imputer.statistics_}")

# Export
joblib.dump(imputer, 'bluecradle_imputer.pkl')
print("✅ bluecradle_imputer.pkl exported successfully.")