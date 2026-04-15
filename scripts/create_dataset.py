from pathlib import Path

import pandas as pd
from Bio import SeqIO

# ---------- CONFIG ----------
# inputs
DATA_DIR = Path("data/raw")
POS_FA = DATA_DIR / "epdnew_400bp.fa"
NEG_FA = DATA_DIR / "epdnew_biasaway_400bp.fa"

# outputs
CSV_OUT = DATA_DIR / "promoter_binary.csv"

WINDOW = 400
SEED = 42
# ----------------------------

def read_fasta_records(path):
    rows = []
    for r in SeqIO.parse(str(path), "fasta"):
        rows.append((r.id, str(r.seq).upper()))
    return rows


def records_to_df(records, label, window):
    return pd.DataFrame(
        [
            {"id": rid, "sequence": seq[:window], "label": label}
            for rid, seq in records
            if len(seq) >= window and "N" not in seq[:window]
        ]
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Building CSV...")
    pos_records = read_fasta_records(POS_FA)
    neg_records = read_fasta_records(NEG_FA)

    pos_df = records_to_df(pos_records, label=1, window=WINDOW)
    print(f"There are {len(pos_df)} positive samples.")
    neg_df = records_to_df(neg_records, label=0, window=WINDOW)
    print(f"There are {len(neg_df)} negative samples.")

    if pos_df.empty:
        raise RuntimeError("No valid positive sequences after filtering.")
    if neg_df.empty:
        raise RuntimeError("No valid negative sequences after filtering.")

    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=n, random_state=SEED)
    neg_df = neg_df.sample(n=n, random_state=SEED)

    df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    df.to_csv(CSV_OUT, index=False)

    print(f"Saved: {CSV_OUT} with shape {df.shape}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
