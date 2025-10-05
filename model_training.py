from pathlib import Path

import numpy as np
import pandas as pd
import time
import collections
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier
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

# Load and tidy the engineered feature sheet
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

# Stratified train/test split and class weights
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
sample_weight_train = compute_sample_weight(class_weight='balanced', y=y_train)

# Model pipeline (Gradient Boosting)
gb_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('gb', ThresholdClassifier(
        base_estimator=GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
        ),
        thresholds={3: 0.25},
        reject_margin=0.1
    ))
])

if __name__ == "__main__":
    start = time.time()
    gb_pipeline.fit(X_train, y_train, gb__sample_weight = sample_weight_train)
    print(f"Training completed in {time.time()-start:.2f}s")

    joblib.dump(gb_pipeline, "gradient_boosting_ctg_model.pt")
    print("Model saved as gradient_boosting_ctg_model.pt")

    interpret_dir = Path("misc/interpretability")
    interpret_dir.mkdir(parents=True, exist_ok=True)

    gb_model = gb_pipeline.named_steps['gb'].estimator_
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    feature_importances.to_csv(interpret_dir / "feature_importance.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances (Gradient Boosting)")
    plt.tight_layout()
    plt.savefig(interpret_dir / "feature_importance_top10.png", dpi=300)
    plt.close()
    print("Feature importance saved to misc/interpretability/")