# Loan Default Prediction

This repo contains an end-to-end pipeline to train and evaluate multiple classifiers on the **UCI “Default of Credit Card Clients (Taiwan)”** dataset and **Lending Club Loan Dataset** dataset, including:

- Logistic Regression (Elastic-Net, CV)
- Decision Tree (with isotonic calibration)
- XGBoost (tuned)
- LightGBM (tuned)
- CatBoost (tuned)

The notebook performs:
1) data loading and dataset-specific fixes,  
2) preprocessing with a shared `ColumnTransformer`,  
3) model training & evaluation with ROC-AUC / PR-AUC / F1 / Recall@TopK,  
4) best-F1 threshold search, and  
5) saving all artifacts (models, preprocessor, feature order, probabilities, curves, metrics, and environment info).

---
## 📁 Repository Structure

```text
.
├── data/
│   └── clients.xls                  # Dataset (UCI Credit Card Default)
│
├── model_runs/
│   ├── CatBoost/
│   ├── Decision Trees/
│   ├── LightGBM/
│   ├── Logistic Regression/
│   ├── XGBoost/
│   └── Lending Club Dataset/
│       ├── best_threshold.joblib        # Model artifact
│       ├── categorical_features.joblib  # Preprocessing artifact
│       ├── feature_columns.joblib       # Preprocessing artifact
│       ├── imputation_values.joblib     # Preprocessing artifact
│       ├── label_encoders.joblib        # Preprocessing artifact
│       ├── lgbm_calibrated_model.joblib # Model artifact
│       └── shap_explainer.joblib        # Model artifact
│
├── src/
│   ├── app.py                       # Main application (Streamlit) for Lending Club dataset
│   ├── lending_club_models.ipynb    # Notebook for Lending Club dataset
│   └── UCI_Combined.ipynb           # Notebook for UCI dataset
│
└── README.md
```
Each folder inside `model_runs/` contains:
- `*_model.joblib` — trained model  
- `*_preprocessor.joblib` — fitted ColumnTransformer  
- `*_feature_names.json` — list of features after preprocessing  
- `*_metrics.json` — metrics (AUC, F1, Precision, Recall, etc.)  
- `*_y_prob.npy` — predicted probabilities  
- `*_roc_curve.npz`, `*_pr_curve.npz` — curve data  
- `*_split_info.json`, `*_env_versions.json` — metadata & reproducibility info  

---

## 🧾 Dataset

- **Source:** [UCI Default of Credit Card Clients Dataset]([https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients))
- **File path:** `data/clients.xls`
- **Target column:** `default_payment_next_month`
---

## ▶️ How to Use

You can either:

### ✅ **Use the saved models**
Simply load the pre-trained models available in the `model_runs` folder.  
Each subfolder (e.g., `xgboost/`, `lightgbm/`) contains all the necessary model artifacts.

or

### 🔁 **Retrain the models**
1. Upload the `src/UCI_Combined.ipynb` notebook to **Google Colab**.  
2. Upload the dataset file (`clients.xls`) to your **Google Drive → MyDrive**.  
3. Update the dataset path in the notebook if needed:
   ```python
   DATA_PATH = "/content/drive/MyDrive/clients.xls"
# For Lending Club dataset:
1.  Downloads the "lending-club" dataset from Kaggle using a shell command.
2. Unzips the downloaded file using a Python script.

### Run the Web Application
The src/app.py file runs a web interface using Streamlit to interact with the saved models.

### ⚠️ Common Issue: LightGBM / pycparser Error

While running the LightGBM training cell in Google Colab, you may encounter an **AttributeError** or import failure caused by a corrupted installation of `pycparser`.

**Fix:**
```python
!pip uninstall -y pycparser lightgbm
!pip install pycparser lightgbm

