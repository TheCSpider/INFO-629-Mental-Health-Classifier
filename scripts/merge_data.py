import pandas as pd
import os

print("Current working directory:")
print(os.getcwd())

# ---------------------------------
# Load the first dataset 
# ---------------------------------
data1 = pd.read_csv("../data/mental_health_prediction/cleanData.csv")
data1 = data1.drop(columns=['index'])  # Drop the unnecessary index column
print("Data1 status: ", data1['status'].unique())

# ---------------------------------
# Load the second dataset 
# ---------------------------------
data2 = pd.read_csv("../data/mental_health_text_classification_dataset/mental_health_unbalanced.csv")
data2 = data2.drop(columns=['Unique_ID'])  # Drop the unnecessary Unique_ID column
data2 = data2.rename(columns={'text': 'statement'})  # Rename columns to match data1
print("Data2 status: ", data2['status'].unique())


# Remove the extra status values from data1 to match data2
data1 = data1[data1['status'].isin(data2['status'].unique())]

print("----------------------------------")
print("Data1 status: ", data1['status'].unique())
print("Data2 status: ", data2['status'].unique())
print("----------------------------------")

# ----------------------------------------------------
# Combine the two datasets and save to a new CSV file
# ----------------------------------------------------
combined = pd.concat([data1, data2])
combined.to_csv("../processed/combined.csv", index=False)

print("Done merging!")
