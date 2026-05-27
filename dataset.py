import pandas as pd

df = pd.read_csv("data/fer2013/fer2013.csv")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())

