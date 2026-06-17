"""
╔══════════════════════════════════════════════════════════════════════╗
║         CreditWise Loan Approval System  —  KNN Model              ║
║         SecureTrust Bank | ML Engineer Project                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  CreditWise Loan System — KNN Classifier")
print("=" * 65)

df = pd.read_csv('loan_approval_data.csv')   # ← update path as needed
print(f"\n[1] Dataset loaded  ➜  {df.shape[0]} rows, {df.shape[1]} columns")
print(f"    Target split  ➜  Approved: {(df['Loan_Approved']=='Yes').sum()}"
      f"  |  Rejected: {(df['Loan_Approved']=='No').sum()}")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
df.dropna(inplace=True)
df.drop(columns=['Applicant_ID'], inplace=True)
print(f"\n[2] After dropping nulls  ➜  {df.shape[0]} rows remain")

# Encode categoricals
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Loan_Approved')
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
df['Loan_Approved'] = (df['Loan_Approved'] == 'Yes').astype(int)
print(f"    Encoded columns: {cat_cols}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN / TEST SPLIT  +  SCALING
# ══════════════════════════════════════════════════════════════════════
X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\n[3] Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — FIND OPTIMAL K  (5-Fold Cross-Validation)
# ══════════════════════════════════════════════════════════════════════
print("\n[4] Searching for best K  (K = 1 … 30) …")
k_scores = {}
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_sc, y_train, cv=5,
                            scoring='accuracy').mean()
    k_scores[k] = score

best_k = max(k_scores, key=k_scores.get)
print(f"    Best K = {best_k}  |  CV Accuracy = {k_scores[best_k]:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN FINAL KNN MODEL
# ══════════════════════════════════════════════════════════════════════
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_sc, y_train)

y_pred = knn_final.predict(X_test_sc)
y_prob = knn_final.predict_proba(X_test_sc)[:, 1]

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION
# ══════════════════════════════════════════════════════════════════════
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)
tp, tn = cm[1,1], cm[0,0]
fp, fn = cm[0,1], cm[1,0]

precision   = tp / (tp + fp) if (tp + fp) else 0
recall      = tp / (tp + fn) if (tp + fn) else 0
f1          = 2*precision*recall / (precision+recall) if (precision+recall) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n" + "=" * 65)
print("  MODEL PERFORMANCE")
print("=" * 65)
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  ROC-AUC     : {auc:.4f}")
print(f"  Precision   : {precision*100:.2f}%")
print(f"  Recall      : {recall*100:.2f}%")
print(f"  F1-Score    : {f1*100:.2f}%")
print(f"  Specificity : {specificity*100:.2f}%")
print("\n  Confusion Matrix:")
print(f"          Pred-Rejected  Pred-Approved")
print(f"  Actual-Rejected   {tn:>4}          {fp:>4}")
print(f"  Actual-Approved   {fn:>4}          {tp:>4}")
print("\n" + classification_report(y_test, y_pred,
      target_names=['Rejected','Approved']))

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — PREDICT FOR A NEW APPLICANT  (demo)
# ══════════════════════════════════════════════════════════════════════
def predict_loan(applicant_dict: dict) -> str:
    """
    Pass a dict with keys matching the feature columns.
    Categorical values must already be label-encoded integers.
    """
    row = pd.DataFrame([applicant_dict])[X.columns]
    row_sc = scaler.transform(row)
    decision = knn_final.predict(row_sc)[0]
    prob     = knn_final.predict_proba(row_sc)[0][1]
    label    = "✅ APPROVED" if decision == 1 else "❌ REJECTED"
    return f"{label}  (approval probability: {prob*100:.1f}%)"

# Example — numeric/encoded demo applicant
sample = {col: float(X_train.iloc[0][col]) for col in X.columns}
print("=" * 65)
print("  SAMPLE PREDICTION")
print("=" * 65)
print(f"  Result: {predict_loan(sample)}")
print("=" * 65)
