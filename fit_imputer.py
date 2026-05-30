import numpy as np
import joblib
from sklearn.impute import SimpleImputer

# Load MAL-ED tensors
data = np.load('data/prepared_data/maled_tensors.npz', allow_pickle=True)
X_train_seq = data['X_train_seq']  # shape: (560, 6, 8)

# Load infant static data to get birth_weight_kg and birth_length_cm
X_train_static = data['X_train_static']  # shape: (560, 3) — sex, birth_weight_kg, birth_length_cm

print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"X_train_static shape: {X_train_static.shape}")

n_children, n_timesteps, n_seq_features = X_train_seq.shape

# Rebuild flattened array with 10 features matching scaler order:
# age_in_days, weight_kg, height_cm, MUAC_mm, birth_weight_kg, birth_length_cm,
# days_since_last_visit, weight_delta_kg, WHZ, whz_velocity

rows = []
for i in range(n_children):
    birth_weight = X_train_static[i, 1]  # birth_weight_kg
    birth_length = X_train_static[i, 2]  # birth_length_cm
    for t in range(n_timesteps):
        seq_row = X_train_seq[i, t, :]  # 8 features
        # Insert birth_weight and birth_length after MUAC_mm (index 3)
        # Original order: age_in_days, weight_kg, height_cm, MUAC_mm,
        #                 days_since_last_visit, weight_delta_kg, WHZ, whz_velocity
        # Target order:   age_in_days, weight_kg, height_cm, MUAC_mm,
        #                 birth_weight_kg, birth_length_cm,
        #                 days_since_last_visit, weight_delta_kg, WHZ, whz_velocity
        new_row = np.concatenate([
            seq_row[:4],                          # age, weight, height, MUAC
            [birth_weight, birth_length],         # birth stats
            seq_row[4:]                           # days_since, weight_delta, WHZ, whz_velocity
        ])
        rows.append(new_row)

seq_flat = np.array(rows, dtype=np.float32)
print(f"Combined flat shape: {seq_flat.shape}")

# Replace padding value -999.0 with NaN
seq_flat[seq_flat == -999.0] = np.nan

print(f"NaN count per feature: {np.sum(np.isnan(seq_flat), axis=0)}")

# Fit imputer on 10 features
imputer = SimpleImputer(strategy='mean')
imputer.fit(seq_flat)

print(f"Imputer fitted on {len(imputer.statistics_)} features.")
print(f"Feature means: {imputer.statistics_}")

# Export
joblib.dump(imputer, 'models/bluecradle_imputer.pkl')
print("✅ bluecradle_imputer.pkl exported successfully.")