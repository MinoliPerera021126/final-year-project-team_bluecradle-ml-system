# BlueCradle Machine Learning System

This repository contains the machine learning training and evaluation pipeline for **BlueCradle**, a regional infant health monitoring system designed for Sri Lankan public healthcare. 

The core intelligence of this system is a longitudinal forecasting model that predicts a child's trajectory toward Moderate Acute Malnutrition (MAM) or Severe Acute Malnutrition (SAM) before clinical onset, prioritizing early warning over standard cross-sectional diagnosis.

---

## 🧠 Model Architecture
The final selected model is a **Hybrid LSTM + Feedforward Neural Network** built with TensorFlow/Keras.
* **Sequential Branch (LSTM):** Processes longitudinal visit data (e.g., historical weights, heights).
* **Static Branch (Dense):** Processes fixed demographic data (e.g., birth weight, sex).
* **Selection Rationale:** Deep evaluation proved that while classical trees (Random Forest, XGBoost) achieved near-perfect accuracy on cross-sectional data due to target leakage (WHZ/MUAC definitions), the LSTM successfully models true longitudinal forecasting, which is critical for preventative intervention.

---

## 📂 Repository Structure
```text
bluecradle-ml-system/
├── data/                   # Raw and prepared datasets (ignored in version control)
├── mlflow/                 # Local SQLite tracking DB and artifacts (ignored in version control)
├── models/                 # Final exported artifacts for backend inference
│   ├── bluecradle_lstm_v1.keras
│   ├── bluecradle_imputer.pkl
│   └── bluecradle_scaler.pkl
├── notebooks/              # The core machine learning pipeline
│   ├── 1_data_preparation.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_model_evaluation.ipynb
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Setup & Execution
### 1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. MLflow Tracking UI
This project uses a sandboxed local MLflow environment to track all hyperparameters, metrics, and evaluation curves (ROC, Precision-Recall). To view the training history, start the server from the root directory:
```bash
cd mlflow
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts
```
Then navigate to `http://127.0.0.1:5000` in your browser.

### 3. Pipeline Execution
To retrain the model from scratch, execute the Jupyter notebooks sequentially:
1. `1_data_preparation.ipynb`: Cleans data, handles missing values (MICE), balances classes (SMOTE), and engineers features.
2. `2_model_training.ipynb `: Trains baseline trees and the deep learning architecture.
3. `3_model_evaluation.ipynb`: Generates deep evaluation metrics, compares models, and exports the winning artifacts.

---

## 🔌 Django Backend Integration Specification
To deploy this model into the BlueCradle backend API, the Django `views.py` endpoint must strictly adhere to the following schema:
1. Load Artifacts Globally: Load the `.keras` model, the `.pkl` imputer, and the `.pkl` scaler once when the server starts to minimize inference latency.
2. Preprocess Payload: Incoming JSON data must be flattened. Missing values must be filled using `bluecradle_imputer.pkl`. Continuous features must be standardized using `bluecradle_scaler.pkl`.
3. Inference & Mapping: The LSTM will return a 3D probability array. The backend must extract `np.argmax(predictions)` and map it to clinical definitions:
- `0` -> Normal (Standard regional monitoring)
- `1` -> MAM (Alert PHM / Midwife)
- `2` -> SAM (Immediate clinical intervention required)
---