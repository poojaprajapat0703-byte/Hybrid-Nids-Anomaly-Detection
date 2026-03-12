############################################################
# Hybrid Network Intrusion Detection System (NIDS)
# Author: Pooja Prajapat
############################################################

"""
Pipeline Overview

Phase 1 — Data Preparation
Phase 1 — Feature Engineering
Phase 1 — Binary Classification Models
Phase 1 — Multiclass Attack Classification

Phase 2 — Anomaly Detection (Isolation Forest)

Phase 3 — XGBoost Attack Classifier

Phase 4 — Real-time NIDS Detection Pipeline
"""

############################################################
# IMPORT LIBRARIES
############################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Train test split
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Machine Learning Models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Handle class imbalance #
from imblearn.over_sampling import SMOTE

# Gradient Boosting #

from xgboost import XGBClassifier
from xgboost import plot_importance

# Save Models #

import joblib

############################################################
# PHASE 1 — LOAD DATASETS
############################################################

print("Loading datasets...")

back = pd.read_csv("data/Data_of_Attack_Back.csv")
buffer_overflow = pd.read_csv("data/Data_of_Attack_Back_BufferOverflow.csv")
ftpwrite = pd.read_csv("data/Data_of_Attack_Back_FTPWrite.csv")
guesspassword = pd.read_csv("data/Data_of_Attack_Back_GuessPassword.csv")
neptune = pd.read_csv("data/Data_of_Attack_Back_Neptune.csv")
nmap = pd.read_csv("data/Data_of_Attack_Back_NMap.csv")
normal = pd.read_csv("data/Data_of_Attack_Back_Normal.csv")
portsweep = pd.read_csv("data/Data_of_Attack_Back_PortSweep.csv")
rootkit = pd.read_csv("data/Data_of_Attack_Back_RootKit.csv")
satan = pd.read_csv("data/Data_of_Attack_Back_Satan.csv")
smurf = pd.read_csv("data/Data_of_Attack_Back_Smurf.csv")

datasets = {
    "back": back,
    "buffer_overflow": buffer_overflow,
    "ftpwrite": ftpwrite,
    "guesspassword": guesspassword,
    "neptune": neptune,
    "nmap": nmap,
    "normal": normal,
    "portsweep": portsweep,
    "rootkit": rootkit,
    "satan": satan,
    "smurf": smurf
}

for name,data in datasets.items():
    print(name, data.shape)

############################################################
# DATA PREPARATION
############################################################

for data in datasets.values():
    data.columns = data.columns.str.strip()

back["attack_type"] = "back"
buffer_overflow["attack_type"] = "buffer_overflow"
ftpwrite["attack_type"] = "ftpwrite"
guesspassword["attack_type"] = "guesspassword"
neptune["attack_type"] = "neptune"
nmap["attack_type"] = "nmap"
normal["attack_type"] = "normal"
portsweep["attack_type"] = "portsweep"
rootkit["attack_type"] = "rootkit"
satan["attack_type"] = "satan"
smurf["attack_type"] = "smurf"

df = pd.concat(list(datasets.values()), axis=0)
df.reset_index(drop=True, inplace=True)

print("Merged dataset shape:", df.shape)

############################################################
# CREATE BINARY LABEL
############################################################

df["label"] = df["attack_type"].apply(lambda x: 0 if x=="normal" else 1)
print(df["label"].value_counts())

############################################################
# DATA CLEANING
############################################################

print("Missing values:", df.isnull().sum().sum())
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

############################################################
# CLASS DISTRIBUTION
############################################################

print(df["label"].value_counts(normalize=True)*100)

############################################################
# EDA
############################################################

print(df.info())
print(df.describe())

print(df["attack_type"].value_counts())

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df["src_bytes"].hist(bins=50)
plt.title("Distribution of src_bytes")
plt.show()

############################################################
# TRAIN TEST SPLIT
############################################################

X = df.drop(["attack_type","label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

print(X_train.shape, X_test.shape)

############################################################
# FEATURE SCALING
############################################################

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############################################################
# HANDLE CLASS IMBALANCE
############################################################

X_train_scaled = np.nan_to_num(X_train_scaled)
y_train = y_train.values.ravel()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(pd.Series(y_train_smote).value_counts())

############################################################
# MODEL TRAINING
############################################################

log_model = LogisticRegression(max_iter=1000,class_weight="balanced")
log_model.fit(X_train_smote,y_train_smote)
y_pred_log = log_model.predict(np.nan_to_num(X_test_scaled))

print("Logistic Accuracy:", accuracy_score(y_test,y_pred_log))

############################################################
# DECISION TREE
############################################################

dt_model = DecisionTreeClassifier(max_depth=10,class_weight="balanced")
dt_model.fit(X_train_smote,y_train_smote)
y_pred_dt = dt_model.predict(np.nan_to_num(X_test_scaled))

print("Decision Tree Accuracy:", accuracy_score(y_test,y_pred_dt))

############################################################
# RANDOM FOREST
############################################################

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train_smote,y_train_smote)
y_pred_rf = rf_model.predict(np.nan_to_num(X_test_scaled))
print("Random Forest Accuracy:", accuracy_score(y_test,y_pred_rf))

############################################################
# MULTICLASS CLASSIFIER
############################################################

y_multi = df["attack_type"]
X_multi = df.drop(["attack_type","label"], axis=1)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi,y_multi,test_size=0.2,stratify=y_multi,random_state=42
)

X_train_m = pd.get_dummies(X_train_m)
X_test_m = pd.get_dummies(X_test_m)
X_train_m, X_test_m = X_train_m.align(X_test_m, join="left", axis=1, fill_value=0)
scaler_m = StandardScaler()
X_train_m_scaled = scaler_m.fit_transform(X_train_m)
X_test_m_scaled = scaler_m.transform(X_test_m)
dt_multi = DecisionTreeClassifier(max_depth=12,class_weight="balanced")
dt_multi.fit(X_train_m_scaled,y_train_m)
y_pred_multi = dt_multi.predict(X_test_m_scaled)
print("Multiclass Accuracy:", accuracy_score(y_test_m,y_pred_multi))

############################################################
# PHASE 2 — ANOMALY DETECTION
############################################################

X_anomaly = df.drop(["attack_type","label"], axis=1)
X_anomaly = X_anomaly.fillna(0)
normal_index = df["label"] == 0
X_normal = X_anomaly[normal_index]
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

iso_model.fit(X_normal)
anomaly_predictions = iso_model.predict(X_anomaly)
df["anomaly_flag"] = anomaly_predictions
print(df["anomaly_flag"].value_counts())

############################################################
# PHASE 3 — XGBOOST CLASSIFIER
############################################################

X_xgb = df.drop("label", axis=1)
X_xgb = pd.get_dummies(X_xgb)
X_xgb = X_xgb.drop(columns=[col for col in X_xgb.columns if "attack_type" in col])
y_xgb = df["label"]
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb,y_xgb,test_size=0.2,random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1
)

xgb_model.fit(X_train_xgb,y_train_xgb)

y_pred_xgb = xgb_model.predict(X_test_xgb)

print("XGBoost Accuracy:", accuracy_score(y_test_xgb,y_pred_xgb))

############################################################
# FEATURE IMPORTANCE
############################################################

plt.figure(figsize=(10,6))
plot_importance(xgb_model,max_num_features=10)
plt.title("Top Important Features")
plt.show()

############################################################
# PHASE 4 — NIDS PIPELINE
############################################################

def nids_detection(sample):

    sample_iso = sample.reindex(columns=X_normal.columns, fill_value=0)

    anomaly_result = iso_model.predict(sample_iso)

    if anomaly_result[0] == -1:
        return "⚠️ Unknown Anomaly Detected"

    attack_result = xgb_model.predict(sample)

    if attack_result[0] == 0:
        return "Normal Traffic"

    attack_type = dt_multi.predict(sample)

    return f"Attack Detected: {attack_type[0]}"

############################################################
# TEST PIPELINE
############################################################

sample_traffic = X_test_xgb.iloc[[0]]

print(nids_detection(sample_traffic))

############################################################
# SIMULATE PACKETS
############################################################

for i in range(10):

    packet = X_test_xgb.iloc[[i]]
    result = nids_detection(packet)
    print(f"Packet {i+1} → {result}")

############################################################
# SAVE MODELS
############################################################

joblib.dump(xgb_model,"xgboost_model.pkl")
joblib.dump(iso_model,"isolation_forest.pkl")
joblib.dump(dt_multi,"multiclass_model.pkl")

print("Models saved successfully")