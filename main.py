import pandas as pd
import numpy as np
import json
import time
import joblib
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read dataset in chunks
def read_csv_in_chunks(file_path, chunksize=10000):
    logging.info("Loading dataset...")
    chunks = [chunk for chunk in pd.read_csv(file_path, chunksize=chunksize)]
    df = pd.concat(chunks, ignore_index=True)
    logging.info(f"Dataset loaded with {len(df)} rows.")
    return df

data = read_csv_in_chunks('merged_data.csv')

# Required columns
required_columns = ['input_addresses', 'output_addresses', 'input_values', 'output_values']
missing = [col for col in required_columns if col not in data.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Validate and filter malformed list strings
def valid_list_string(x):
    return pd.notnull(x) and isinstance(x, str) and x.startswith('[') and x.endswith(']')

for col in required_columns:
    data = data[data[col].apply(valid_list_string)]

# Fast and safe list parsing
def fast_parse(val):
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []

for col, new_col in [('input_addresses', 'input_addr'),
                     ('output_addresses', 'output_addr'),
                     ('input_values', 'input_value'),
                     ('output_values', 'output_value')]:
    start = time.time()
    logging.info(f"Parsing {col}...")
    data[new_col] = data[col].map(fast_parse)
    logging.info(f"{col} parsed in {time.time() - start:.2f}s")

# Parse datetime
if 'time' in data.columns:
    logging.info("Parsing datetime column...")
    data['datetime'] = pd.to_datetime(data['time'], unit='s', errors='coerce')

# Feature engineering
logging.info("Creating features...")
data['num_input_addresses'] = data['input_addr'].apply(len)
data['num_output_addresses'] = data['output_addr'].apply(len)
data['total_input_value'] = data['input_value'].apply(sum)
data['total_output_value'] = data['output_value'].apply(sum)

data['sending_amount'] = np.where(data['transaction_type'] == 'sent', data['total_output_value'], 0)
data['receiving_amount'] = np.where(data['transaction_type'] == 'received', data['total_input_value'], 0)

data['balance'] = data['receiving_amount'] - data['sending_amount']
data['size_ratio'] = data['num_input_addresses'] / (data['num_output_addresses'] + 1)
data['inflow_outflow_ratio'] = (data['receiving_amount'] + 1) / (data['sending_amount'] + 1)

data['primary_address'] = data['address']
data['transaction'] = data['transaction_type']
data['total_transactions'] = data.groupby('primary_address')['transaction'].transform('count')

BTC_TO_USD = 30000
data['usd_value'] = np.where(data['transaction'] == 'sent', data['sending_amount'], data['receiving_amount']) * BTC_TO_USD

def extract_digit_scale(x):
    return 0 if x == 0 else int(np.floor(np.log10(abs(x))))

data['digit_scale'] = data['usd_value'].apply(extract_digit_scale)

def digit_frequency(df, tx_type):
    filtered = df[df['transaction'] == tx_type]
    freq = filtered.groupby(['primary_address', 'digit_scale']).size().unstack(fill_value=0)
    freq.columns = [f'f_{tx_type}_10^{col}' for col in freq.columns]
    return freq

f_spent = digit_frequency(data, 'sent')
f_received = digit_frequency(data, 'received')
data = data.merge(f_spent, on='primary_address', how='left')
data = data.merge(f_received, on='primary_address', how='left')

def compute_ratios(df):
    grp = df.groupby('primary_address')['transaction'].value_counts(normalize=True).unstack().fillna(0)
    grp.columns = [f'r_{col}' for col in grp.columns]
    return grp

ratios = compute_ratios(data)
data = data.merge(ratios, on='primary_address', how='left')

def mean_io_in_spent(df):
    spent = df[df['transaction'] == 'sent']
    grp = spent.groupby('primary_address').agg({'num_input_addresses': 'mean', 'num_output_addresses': 'mean'})
    grp.rename(columns={
        'num_input_addresses': 'N_inputs',
        'num_output_addresses': 'N_outputs'
    }, inplace=True)
    return grp

mean_io = mean_io_in_spent(data)
data = data.merge(mean_io, on='primary_address', how='left')

def payback_ratio(row):
    input_set, output_set = set(row['input_addr']), set(row['output_addr'])
    union = input_set | output_set
    return 0 if len(union) == 0 else len(input_set & output_set) / len(union)

data['r_payback'] = data.apply(payback_ratio, axis=1)

if 'datetime' in data.columns:
    freq_per_day = data.groupby('primary_address').agg({
        'datetime': lambda x: x.count() / ((x.max() - x.min()).days + 1 if x.max() != x.min() else 1)
    })
    freq_per_day.columns = ['f_TX']
    data = data.merge(freq_per_day, on='primary_address', how='left')

# Encode target class
if 'class' not in data.columns:
    raise KeyError("Missing 'class' column in the dataset.")

le = LabelEncoder()
data['class_encoded'] = le.fit_transform(data['class'])

# Define feature list
features = [
    'num_input_addresses', 'num_output_addresses', 'total_input_value', 'total_output_value',
    'sending_amount', 'receiving_amount', 'balance', 'size_ratio', 'inflow_outflow_ratio',
    'total_transactions', 'N_inputs', 'N_outputs', 'r_payback'
]

# Conditionally add f_TX if it was created
if 'f_TX' in data.columns:
    features.append('f_TX')

# Add ratio columns if present
features += [col for col in ['r_sent', 'r_received', 'r_coinbase'] if col in data.columns]

# Add digit frequency columns
features += [col for col in data.columns if col.startswith('f_sent_10^') or col.startswith('f_received_10^')]

# Filter for existing features
valid_features = [f for f in features if f in data.columns]
missing_features = set(features) - set(valid_features)
if missing_features:
    logging.warning(f"The following features are missing and will be skipped: {missing_features}")

X = data[valid_features].fillna(0)
y = data['class_encoded']

logging.info("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
logging.info("Training models...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42)

rf.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)

for name, model in [('Random Forest', rf), ('LightGBM', lgbm), ('XGBoost', xgb), ('CatBoost', cat)]:
    y_pred = model.predict(X_test)
    logging.info(f"{name} Performance:")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    logging.info(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Ensemble model
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('lgb_xgb', VotingClassifier(estimators=[('lgbm', lgbm), ('xgb', xgb)], voting='soft')),
    ('cat', cat)
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

logging.info("Ensemble Model Performance:")
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
logging.info(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
logging.info(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

joblib.dump(ensemble, 'ensemble_model.pkl')
