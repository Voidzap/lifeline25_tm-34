
"""
CTG (Cardiotocography) Cleaning & EDA Template
----------------------------------------------
Usage:
  python ctg_cleaning_template.py --file /path/to/CTG.xls  # or ctg.csv

What it does:
  1) Loads data from Excel (.xls) or CSV.
  2) Basic sanity checks: shapes, dtypes, duplicate rows.
  3) Confirms missing values (UCI version claims none, but we verify).
  4) Encodes labels (chooses "NSP" 3-class target by default), with an option to use "CLASS".
  5) Train/validation/test split (stratified).
  6) Scaling (StandardScaler) on numeric features (fit on train only).
  7) Saves:
     - cleaned_full.csv (all rows with selected target & numeric features)
     - ctg_train.csv, ctg_val.csv, ctg_test.csv
     - scaler.pkl (fitted StandardScaler)
  8) Visuals:
     - Histograms per feature
     - Correlation heatmap (matplotlib only)
     - Scatter-matrix (pandas.plotting.scatter_matrix) for a small subset
  9) Prints class balance & simple summary stats.

Notes:
  - Pairplots are typically done with seaborn; we avoid seaborn per your constraints.
  - If you want to switch to the 10-class "CLASS" label, use --target CLASS.
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_FEATURES = [
    # Core CTG features from UCI (order doesn't matter)
    "LB","AC","FM","UC","DL","DS","DP",
    "ASTV","MSTV","ALTV","MLTV",
    "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"
]

def load_ctg(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        # Try common fallbacks in current dir
        for guess in ["CTG.xls", "CTG.csv", "ctg.csv", "Cardiotocography.csv", "Cardiotocography.xls"]:
            g = Path(guess)
            if g.exists():
                path = g
                break
    if not path.exists():
        raise FileNotFoundError(f"Could not find file at {path}. Please pass --file /path/to/CTG.xls or place CTG.xls in this folder.")
    if path.suffix.lower() in [".xls", ".xlsx"]:
        # The UCI CTG.xls has sheet "Raw Data" or similar; try first sheet by default.
        try:
            df = pd.read_excel(path, sheet_name=0)
        except Exception as e:
            raise RuntimeError(f"Failed reading Excel: {e}")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .xls/.xlsx or .csv")
    # Some distributions include unnamed index columns—drop them
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace, unify column casing for robustness
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def choose_target(df: pd.DataFrame, target: str) -> str:
    cols = set(df.columns)
    if target == "AUTO":
        # Prefer 3-class 'NSP' (Normal=S, Suspect=S, Pathologic=P) if present, else 'CLASS'
        if "NSP" in cols:
            return "NSP"
        elif "CLASS" in cols:
            return "CLASS"
        else:
            # Some variants store it as "CLASS" or "Nsp" etc.
            guess = [c for c in df.columns if c.upper() in {"NSP","CLASS"}]
            if guess:
                return guess[0]
            raise KeyError("Could not find NSP or CLASS in columns. Available: " + ", ".join(df.columns))
    else:
        if target not in cols:
            raise KeyError(f"Requested target '{target}' not in columns: {', '.join(df.columns)}")
        return target

def basic_checks(df: pd.DataFrame):
    print("\n=== Basic Info ===")
    print(df.shape)
    print(df.dtypes.value_counts())
    print("\nHead:\n", df.head(3))
    # Duplicates
    dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup}")
    # Missing check
    miss = df.isna().sum().sort_values(ascending=False)
    if miss.any():
        print("\nMissing values (non-zero):")
        print(miss[miss>0])
    else:
        print("\nNo missing values detected (nice!).")

def get_features(df: pd.DataFrame) -> list:
    # Use DEFAULT_FEATURES if present; otherwise fall back to all numeric except target
    present = [f for f in DEFAULT_FEATURES if f in df.columns]
    return present

def encode_labels(y: pd.Series) -> pd.Series:
    # For NSP: values are typically 1=Normal, 2=Suspect, 3=Pathologic
    # Map to 0/1/2 for ML convenience
    unique = sorted(y.dropna().unique().tolist())
    mapping = {}
    # If labels already 1/2/3 map to 0/1/2; otherwise factorize
    if set(unique) == set([1,2,3]):
        mapping = {1:0, 2:1, 3:2}
        print("NSP mapping: 1->0 (Normal), 2->1 (Suspect), 3->2 (Pathologic)")
        return y.map(mapping)
    else:
        codes, uniques = pd.factorize(y, sort=True)
        print(f"Label factorization mapping: {dict(enumerate(uniques))}")
        return pd.Series(codes, index=y.index)

def correlation_heatmap(df_num: pd.DataFrame, out_path: Path):
    corr = df_num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(corr.values, interpolation='nearest')
    ax.set_xticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def histograms(df_num: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in df_num.columns:
        fig, ax = plt.subplots()
        df_num[col].plot(kind="hist", bins=40, ax=ax)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{col}.png")
        plt.close(fig)

def scatter_matrix_subset(df_num: pd.DataFrame, out_path: Path, max_cols: int = 6):
    # Avoid huge grids; pick up to 6 columns (top variance)
    variances = df_num.var().sort_values(ascending=False)
    cols = variances.index[:max_cols].tolist()
    sm = scatter_matrix(df_num[cols], figsize=(10,10), diagonal='hist')
    # Set titles for diagonals
    for i, c in enumerate(cols):
        sm[i][i].set_title(c)
    plt.suptitle("Scatter matrix (top-variance features)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="CTG.xls", help="Path to CTG.xls or ctg.csv")
    parser.add_argument("--target", type=str, default="AUTO", help="Target column: AUTO | NSP | CLASS | <name>")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./outputs_ctg")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = load_ctg(Path(args.file))
    df_raw = normalize_columns(df_raw)
    basic_checks(df_raw)

    target = choose_target(df_raw, args.target)
    print(f"\nUsing target: {target}")
    # Pick features
    features = get_features(df_raw)
    if len(features) == 0:
        # fallback to all numeric except target
        features = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if target in features:
            features.remove(target)
        print(f"Using ALL numeric columns as features (except target). Count: {len(features)}")
    else:
        print(f"Using default feature set. Count: {len(features)}")

    # Keep only features + target; drop rows with missing in these (shouldn't happen usually)
    cols_needed = features + [target]
    df = df_raw[cols_needed].dropna().copy()

    # Encode labels to 0..K-1
    y = encode_labels(df[target])
    X = df[features].astype(float)

    # Quick class balance
    print("\nClass balance:")
    print(y.value_counts(normalize=False).sort_index())

    # Save full cleaned dataset
    cleaned_full = pd.concat([X, y.rename("label")], axis=1)
    cleaned_full_path = outdir / "cleaned_full.csv"
    cleaned_full.to_csv(cleaned_full_path, index=False)
    print(f"\nSaved cleaned_full.csv -> {cleaned_full_path}")

    # Train/val/test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    # Normalize val split proportion relative to remaining
    val_ratio = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=args.random_state, stratify=y_temp
    )

    # Scaling (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Save splits
    pd.DataFrame(X_train_s, columns=features).assign(label=y_train.values).to_csv(outdir / "ctg_train.csv", index=False)
    pd.DataFrame(X_val_s, columns=features).assign(label=y_val.values).to_csv(outdir / "ctg_val.csv", index=False)
    pd.DataFrame(X_test_s, columns=features).assign(label=y_test.values).to_csv(outdir / "ctg_test.csv", index=False)
    with open(outdir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved splits & scaler to: {outdir}")

    # Visuals
    print("\nGenerating visuals (matplotlib-only)...")
    num_df = X  # numeric features before scaling
    hist_dir = outdir / "histograms"
    histograms(num_df, hist_dir)
    print(f"Histograms saved to: {hist_dir}")

    heatmap_path = outdir / "correlation_heatmap.png"
    correlation_heatmap(num_df, heatmap_path)
    print(f"Correlation heatmap saved to: {heatmap_path}")

    scatter_path = outdir / "scatter_matrix.png"
    scatter_matrix_subset(num_df, scatter_path, max_cols=6)
    print(f"Scatter matrix saved to: {scatter_path}")

    print("\nDone. Next steps for Modeling (Person B):")
    print(" - Load ctg_train/val/test.csv")
    print(" - Train baseline models (LogReg, DT, RF), then XGBoost/LightGBM")
    print(" - Track metrics incl. per-class recall & confusion matrix")
    print(" - Consider class-imbalance strategies (weights/SMOTE)")

main()
