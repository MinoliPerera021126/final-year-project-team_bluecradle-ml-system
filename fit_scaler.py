import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load MAL-ED tensors
data = np.load('data/prepared_data/maled_tensors.npz', allow_pickle=True)
X_train_seq = data['X_train_seq']  # shape: (560, 6, 8)

print(f"X_train_seq shape: {X_train_seq.shape}")

n_children, n_timesteps, n_features = X_train_seq.shape

# Flatten to (3360, 8)
seq_flat = X_train_seq.reshape(-1, n_features).astype(np.float32)

# Replace padding value -999.0 with NaN
seq_flat[seq_flat == -999.0] = np.nan

# Replace NaN with column means before fitting scaler
col_means = np.nanmean(seq_flat, axis=0)
nan_mask = np.isnan(seq_flat)
seq_flat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

print(f"Flat shape: {seq_flat.shape}")

# Fit scaler on 8 features
scaler = StandardScaler()
scaler.fit(seq_flat)

print(f"Scaler fitted on {len(scaler.mean_)} features.")
print(f"Feature means: {scaler.mean_}")

# Export
joblib.dump(scaler, 'models/bluecradle_scaler.pkl')
print("✅ bluecradle_scaler.pkl exported successfully.")