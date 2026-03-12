import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(BASE_DIR, "processed", "combined.csv")
output_path = os.path.join(BASE_DIR, "processed", "cleaned_combined.csv")

df = pd.read_csv(input_path)

allowed_classes =["Normal", "Depression", "Suicidal", "Anxiety"]
df = df[df["status"].isin(allowed_classes)]

print("Original shape:", df.shape)

df.columns = df.columns.str.lower()

if "statement" in df.columns and "text" in df.columns:
    df["text"] = df["statement"].fillna(df["text"])
    df = df.drop(columns=["statement"])

df = df.dropna(subset=["text"])

df = df.drop_duplicates()

df["test"] = df["text"].astype(str).str.lower().str.strip()

print("Cleaned shape:",df.shape)

df.to_csv(output_path, index=False)

print("Cleaning complete!")