import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

inp = Path("data/raw/promoter_binary.csv")
outdir = Path("data/processed/")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp)

# stratified split: 80 train, 10 val, 10 test
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

train_df.to_csv(outdir / "train.csv", index=False)
val_df.to_csv(outdir / "val.csv", index=False)
test_df.to_csv(outdir / "test.csv", index=False)

print("train:", train_df.shape, train_df["label"].value_counts().to_dict())
print("val:", val_df.shape, val_df["label"].value_counts().to_dict())
print("test:", test_df.shape, test_df["label"].value_counts().to_dict())
