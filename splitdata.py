import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("CTG.csv")

df = df.loc[:, ~(
    df.iloc[1].isna() | (df.iloc[1].astype(str).str.strip() == "")
)]

new_headers = [
    "b", "e", 
    "AC", "FM", "UC", "DL", "DS", "DP", "DR",
    "LB", "AC2", "FM2", "UC2", "DL2", "DS2", "DP2",
    "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median",
    "Variance", "Tendency",
    "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP",
    "CLASS", "NSP"
]

df.columns = new_headers

df["NSP"] = pd.to_numeric(df["NSP"], errors="coerce")
df = df[df["NSP"].isin([1,2,3])].copy()

train_idx, test_idx = train_test_split(
    df.index, test_size=0.25, random_state=42, stratify=df["NSP"]
)

train_df = df.loc[train_idx].copy()
test_df  = df.loc[test_idx].copy()

train_df.to_csv("ctg_train_75.csv", index=False)
test_df.to_csv("ctg_test_25.csv", index=False)

print("Train NSP distribution:\n", train_df["NSP"].value_counts(normalize=True).sort_index())
print("\nTest NSP distribution:\n",  test_df["NSP"].value_counts(normalize=True).sort_index())


