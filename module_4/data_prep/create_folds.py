import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load existing train.csv
df = pd.read_csv("segmentation_dataset/train.csv")

# Create fold column
df["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["empty"])):
    df.loc[val_idx, "fold"] = fold

# Save back to file
df.to_csv("segmentation_dataset/train_folds.csv", index=False)

print("Saved segmentation_dataset/train_folds.csv with fold column")
