# Loan Default Prediction

This repo contains an end-to-end pipeline to train and evaluate multiple classifiers on the **UCI “Default of Credit Card Clients (Taiwan)”** dataset, including:

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

.
├── data/  
│   └── clients.xls                # Dataset file (UCI Credit Card Default dataset)  
│  
├── src/  
│   └── UCI_Combined.ipynb         # Main Jupyter notebook (complete pipeline)  
│  
├── model_runs/  
│   ├── logistic_regression/       # Artifacts for Logistic Regression  
│   ├── decision_tree/             # Artifacts for Decision Tree + calibrated model  
│   ├── xgboost/                   # Artifacts for tuned XGBoost  
│   ├── lightgbm/                  # Artifacts for tuned LightGBM  
│   └── catboost/                  # Artifacts for tuned CatBoost  
│  
└── README.md                      # You are here  

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


### ⚠️ Common Issue: LightGBM / pycparser Error

While running the LightGBM training cell in Google Colab, you may encounter an **AttributeError** or import failure caused by a corrupted installation of `pycparser`.

**Fix:**
```python
!pip uninstall -y pycparser lightgbm
!pip install pycparser lightgbm

