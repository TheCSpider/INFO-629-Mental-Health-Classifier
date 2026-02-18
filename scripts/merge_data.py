import pandas as pd
import os

print("Current working directory:")
print(os.getcwd())

data1 = pd.read_csv("data/mental_health_prediction/cleanData.csv")
data2 = pd.read_csv("data/mental_health_text_classification_dataset/mental_health_unbalanced.csv")

combined = pd.concat([data1, data2])
combined.to_csv("processed/combined.csv", index=False)

print("Done merging!")
