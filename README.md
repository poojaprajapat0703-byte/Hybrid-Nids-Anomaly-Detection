# Hybrid Network Intrusion Detection System (NIDS) with Anomaly Detection

## Project Overview

This project implements a **Hybrid Network Intrusion Detection System (NIDS)** using machine learning to detect **malicious network traffic, unknown anomalies, and specific attack types**.

Traditional NIDS systems detect only **known attack signatures**, but modern cyber threats often include **unknown attack patterns**. To address this limitation, this system combines:

* **Unsupervised anomaly detection**
* **Binary attack classification**
* **Multiclass attack identification**

The final system creates a **multi-stage cybersecurity detection pipeline** capable of identifying:

* Normal network traffic
* Known attack types
* Unknown anomalous behavior

---

# System Architecture

The detection pipeline follows a **three-stage hybrid architecture**:

                Incoming Network Traffic
                          │
                          ▼
              +-------------------------+
              |   Isolation Forest      |
              |  (Anomaly Detection)    |
              | Detect Unknown Attacks  |
              +-------------------------+
                          │
                          ▼
              +-------------------------+
              |       XGBoost           |
              | Binary Classification   |
              |  Normal vs Attack       |
              +-------------------------+
                          │
                          ▼
              +-------------------------+
              |     Decision Tree       |
              | Multiclass Classifier   |
              | Identify Attack Type    |
              +-------------------------+
                          │
                          ▼
                   Final Detection

### Stage 1 – Anomaly Detection

Isolation Forest identifies **unusual traffic patterns** that may represent **unknown cyber attacks**.

### Stage 2 – Binary Attack Detection

XGBoost determines whether the traffic is **Normal or Malicious**.

### Stage 3 – Attack Type Classification

If malicious, the Decision Tree classifier identifies the **specific attack type**.

---

# Phase 1 — Data Preparation

## Dataset Description

The dataset consists of multiple network attack categories including:

* Back
* Buffer Overflow
* FTP Write
* Guess Password
* Neptune
* Nmap
* Portsweep
* Rootkit
* Satan
* Smurf
* Normal Traffic

Each dataset was loaded separately and merged into a single dataframe.

---

## Step 1 – Load Datasets

All CSV files representing different attack types were loaded using Pandas.

Example:

```
back = pd.read_csv("Data_of_Attack_Back.csv")
neptune = pd.read_csv("Data_of_Attack_Back_Neptune.csv")
normal = pd.read_csv("Data_of_Attack_Back_Normal.csv")
```

---

## Step 2 – Add Attack Labels

Each dataset was assigned an `attack_type` column.

Example:

```
back["attack_type"] = "back"
neptune["attack_type"] = "neptune"
normal["attack_type"] = "normal"
```

---

## Step 3 – Merge All Datasets

All datasets were concatenated into a single dataframe.

```
df = pd.concat([...], axis=0)
df.reset_index(drop=True, inplace=True)
```

---

## Step 4 – Create Binary Label

A binary classification label was created:

* **0 → Normal Traffic**
* **1 → Attack**

```
df["label"] = df["attack_type"].apply(lambda x: 0 if x=="normal" else 1)
```

---

## Step 5 – Data Cleaning

Data quality checks included:

* Checking missing values
* Removing duplicate rows
* Verifying dataset consistency

---

## Step 6 – Class Imbalance Check

Cybersecurity datasets typically contain **many more attacks than normal samples**.

Class distribution was analyzed using:

```
df["label"].value_counts()
```

---

# Phase 2 — Exploratory Data Analysis (EDA)

Several analyses were performed:

### Dataset Statistics

```
df.describe()
```

### Feature Distribution

Example visualization:

```
df["src_bytes"].hist(bins=50)
```

### Correlation Heatmap

```
sns.heatmap(df.corr())
```

This helped identify **highly correlated features**.

---

# Phase 3 — Train Test Split

Dataset split:

```
Train: 80%
Test: 20%
```

```
X_train, X_test, y_train, y_test = train_test_split(...)
```

Example output:

```
Train: (654040, 82)
Test: (163510, 82)
```

---

# Phase 4 — Feature Engineering

## Feature Scaling

Standardization applied using:

```
StandardScaler()
```

---

## Handling Class Imbalance (SMOTE)

To balance attack vs normal samples:

```
SMOTE(random_state=42)

```

After SMOTE:

```
Attack samples = Normal samples

```

---

## Feature Selection

Highly correlated features (>0.9 correlation) were removed to reduce redundancy.

---

# Phase 5 — Machine Learning Models

Multiple models were trained and compared.

---

## Logistic Regression

Baseline model for binary classification.

Accuracy:

```
~99.7%
```

---

## Decision Tree Classifier

Handles nonlinear relationships effectively.

Accuracy:

```
~99.98%
```

---

## Random Forest Classifier

Ensemble learning model for improved stability.

Accuracy:

```
~99.97%
```

---

## Model Performance Comparison

| Model | Accuracy |
|------|---------|
| Logistic Regression | ~99.7% |
| Decision Tree | ~99.98% |
| Random Forest | ~99.97% |
| XGBoost | ~99.99% |

# Phase 6 — Multiclass Attack Classification

A separate model was trained to identify **specific attack types**.

Model used:

```
DecisionTreeClassifier
```

Multiclass accuracy:

```
~99.7%
```

---

# Phase 7 — Anomaly Detection

Isolation Forest was trained **only on normal traffic**.

Purpose:

Detect **unknown anomalies** that do not match known attack patterns.

Example output:

```
anomaly_flag
1 → Normal
-1 → Anomaly
```

---

# Phase 8 — XGBoost Attack Classifier

XGBoost was trained as the **primary binary attack detection model**.

Advantages:

* Handles complex patterns
* High accuracy
* Fast prediction speed

Accuracy:

```
~99.99%
```

Feature importance visualization was generated using:

```
plot_importance(xgb_model)
```

---

# Phase 9 — Hybrid NIDS Detection Pipeline

A final detection function integrates all models:

```
def nids_detection(sample):
```

Detection sequence:

1. Isolation Forest checks anomaly
2. XGBoost detects attack vs normal
3. Multiclass model identifies attack type

---

# Example Detection Output

```
Packet 1 → Normal Traffic
Packet 2 → Normal Traffic
Packet 3 → Unknown Anomaly Detected
Packet 4 → Normal Traffic
Packet 5 → Unknown Anomaly Detected
```

---

# Model Persistence

Trained models are saved using Joblib:

```
joblib.dump(model, "models/model.pkl")
```

Saved models:

```
models/
├── xgboost_model.pkl
├── isolation_forest.pkl
└── multiclass_model.pkl
```

---

# Project Structure

```
hybrid-nids-anomaly-detection
│
├── data
│
├── models
│   ├── xgboost_model.pkl
│   ├── isolation_forest.pkl
│   └── multiclass_model.pkl
│
├── analysis.ipynb
├── nids_system.py
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SMOTE
* Matplotlib
* Seaborn
* Joblib

---

# Installation

Clone repository:

```
git clone https://github.com/poojaprajapat0703-byte/Hybrid-Nids-Anomaly-Detection.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Run system:

```
python nids_system.py
```

---

# Future Improvements

Possible extensions include:

* Real-time packet capture using **Wireshark / Scapy**
* Cybersecurity monitoring dashboard
* REST API deployment
* Integration with **SIEM systems**
* Deep learning-based intrusion detection

---

# Author

Pooja Prajapat
