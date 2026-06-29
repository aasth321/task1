import pandas as pd
import numpy as np
import seaborn as sns

# 1) Load dataset
def load_titanic():
    try:
        return sns.load_dataset("titanic")
    except:
        return pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

df = load_titanic()

# 2) Handle missing values
def fill_missing(df):
    # Numeric: median
    for col in ["age", "fare"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # Categorical: mode
    for col in ["embarked", "deck", "embark_town"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

df = fill_missing(df)

# 3) Remove outliers (z-score method)
def remove_outliers(df, cols, threshold=3):
    for col in cols:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z < threshold]
    return df

df = remove_outliers(df, ["age", "fare"])
print("\nProcessed Data Preview (first 5 rows)")
print(df.head())

print("\nProcessed Shape (rows, columns)")
print(df.shape)

print("\n Processed Columns")
print(df.columns.tolist())

print("\nSummary Statistics (numeric features) ")
print(df.describe())




