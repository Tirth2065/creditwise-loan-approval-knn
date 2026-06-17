Pipeline (7 steps)

1. Load — 1000 records, 19 features
2. Clean — dropped 50 rows with nulls; removed Applicant_ID
3. Encode — LabelEncoder on all categorical columns; target → 0/1
4. Split — 80% train / 20% test (stratified)
5. Scale — StandardScaler (essential for KNN — it's distance-based)
6. Tune K — 5-fold cross-validation across K = 1…30 → Best K = 11
7. Evaluate — full metrics report

Model             Results
Metric            Score
Best K            11
CV Accuracy       82.5%
Test Accuracy     73.2%
ROC-AUC           0.786
Precision         62%
Recall            43%
F1-Score          51%
