# Bitcoin Transaction Classification using Ensemble Machine Learning

This project builds a **multi-class classification pipeline** to identify Bitcoin address types (e.g., exchange, service, gambling, scam, etc.) based on transaction-level features derived from blockchain data.  
The model uses an **ensemble of Random Forest, LightGBM, XGBoost, and CatBoost classifiers** to achieve high accuracy and robustness.

---

## 🧩 Overview

This script performs the following major steps:

### **1. Data Loading**
- Reads a large merged dataset (`merged_data.csv`) in chunks for memory efficiency.  
- Validates essential columns and filters malformed rows.

### **2. Data Parsing**
- Parses JSON-like list strings for input/output addresses and values.  
- Converts UNIX timestamps to human-readable datetime format.

### **3. Feature Engineering**
Computes transaction-level and behavioral features, including:
- Number of input/output addresses  
- Total input/output values  
- Balance, inflow–outflow ratio, and size ratio  
- Transaction frequencies and digit-based value distributions  
- Payback ratio (self-transaction rate)  
- Mean number of inputs/outputs for sent transactions  
- Transaction ratio per address type

### **4. Model Training**
- Encodes class labels using `LabelEncoder`.  
- Trains four base models: **Random Forest**, **LightGBM**, **XGBoost**, and **CatBoost**.  
- Evaluates each model with **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Confusion Matrix**.

### **5. Ensemble Learning**
- Combines all base models using a **hierarchical soft VotingClassifier**.  
- Achieves strong overall performance with ~**98.69% accuracy**.

### **6. Model Saving**
- Saves the trained ensemble model as `ensemble_model.pkl` using `joblib`.

---

---

## ⚙️ Requirements

Install the following Python dependencies before running the script:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost joblib





**##**
---

![Model Training Overview](https://github.com/user-attachments/assets/6e82fe44-8766-49a2-a106-a3d7ee0f6717)

![Feature Engineering Pipeline](https://github.com/user-attachments/assets/f1cf362d-5a1b-4b58-997a-c985368a918d)


