import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(BASE_DIR, "processed", "cleaned_combined.csv")
df = pd.read_csv(file_path)

print("Data shape:", df.shape)
print("\nColumns:", df.columns)
print("\nClass distribution:")
print(df["status"].value_counts())