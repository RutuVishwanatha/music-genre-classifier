import pandas as pd

df = pd.read_csv('data/features_30_sec.csv')
print("First 5 rows:\n", df.head(), "\n")
print("Shape of dataset:", df.shape, "\n")
print("Columns:\n", df.columns, "\n")
print("Unique genres:", df['label'].unique())
