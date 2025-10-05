from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, thresholds=None, reject_margin=None):
        self.base_estimator = base_estimator
        self.thresholds = thresholds or {}
        self.reject_margin = reject_margin
    
    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = np.array(self.estimator_.classes_)
        if 3 in self.classes_:
            self.cls3_idx = np.where(self.classes_ == 3)[0][0]
        else:
            self.cls3_idx = None
        if 2 in self.classes_:
            self.cls2_idx = np.where(self.classes_ == 2)[0][0]
        else:
            self.cls2_idx = None
        return self
    
    def predict(self, X):
        probs = self.estimator_.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        if 3 in self.thresholds:
            thr = self.thresholds[3]
            override = probs[:, self.cls3_idx] >= thr
            preds_idx[override] = self.cls3_idx
        if self.reject_margin is not None and self.cls3_idx is not None:
            top2_idx = np.argsort(probs, axis=1)[:, -2:]
            top2_vals = np.take_along_axis(probs, top2_idx, axis=1)
            close = (top2_vals[:, 1] - top2_vals[:, 0]) < self.reject_margin
            cls3_in_top2 = np.any(top2_idx == self.cls3_idx, axis=1)
            to_reject = close & cls3_in_top2
            preds_idx[to_reject] = self.cls2_idx
        
        return self.classes_[preds_idx]
    
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

def read_feature_sheet(path: Path, sheet=0):
    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[sheet] if isinstance(sheet, int) else sheet
    return pd.read_excel(xls, sheet_name=sheet_name, header=1)

def tidy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.dropna(axis=0, how='all').dropna(axis=1, how='all')
    cleaned = cleaned.loc[:, ~cleaned.columns.str.contains('^Unnamed', case=False)]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    rename_map = {
        'MSTV': 'mSTV',
        'MLTV': 'mLTV',
        'Variance ': 'Variance',
        'TENDENCY': 'Tendency'
    }
    cleaned = cleaned.rename(columns={k: v for k, v in rename_map.items() if k in cleaned.columns})
    return cleaned

#Load model
model = joblib.load("gradient_boosting_ctg_model.pt")

#Load and tidy test data
ctg_path = Path("data_exploration/cardiotocography/CTG.xls")
raw_features = read_feature_sheet(ctg_path, sheet=1)

# Clean columns, drop leakage, and ensure numerics
sheet2 = tidy_sheet(raw_features)
target_col = 'NSP'
label_leak_cols = ['CLASS', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']
feature_cols = [col for col in sheet2.columns if col not in label_leak_cols + [target_col]]
clean_df = (
    sheet2
    .drop(columns=label_leak_cols, errors='ignore')
    .dropna(axis=0, how='all')
    .drop_duplicates()
)

# Coerce numeric columns and drop rows without labels
clean_df[feature_cols] = clean_df[feature_cols].apply(pd.to_numeric, errors='coerce')
clean_df = clean_df.dropna(subset=[target_col]).copy()
clean_df[target_col] = clean_df[target_col].astype(int)
X = clean_df[feature_cols]
y = clean_df[target_col]

#Predict
y_pred = model.predict(X)
print(classification_report(y, y_pred, digits=3))
print(confusion_matrix(y, y_pred))