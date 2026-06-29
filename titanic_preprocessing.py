import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load the dataset
try:
    df = sns.load_dataset("titanic")
except Exception as e:
    print(f"Could not load seaborn dataset directly: {e}")
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

print("Initial shape:", df.shape)
print("\nInitial info:")
df.info()
print("\nMissing values per column:")
print(df.isna().sum())
print("\nFirst 5 rows:")
print(df.head())

# 2) Handle missing values
# Numeric columns: fill with median
for col in ["age", "fare"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill with mode or a placeholder
for col in ["embarked", "deck", "embark_town"]:
    if col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

# 3) Convert categorical features to numerical
# One-hot encode selected categorical columns
categorical_cols = ["sex", "embarked", "class", "who", "deck", "embark_town"]
encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Keep the target column if present
if "survived" in encoded_df.columns:
    target = encoded_df["survived"]
    features = encoded_df.drop(columns=["survived"])
else:
    target = None
    features = encoded_df

# 4) Normalize/standardize numeric features
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    mean = features[col].mean()
    std = features[col].std()
    if std != 0:
        features[col] = (features[col] - mean) / std

# Recombine features and target
if target is not None:
    processed_df = pd.concat([target, features], axis=1)
else:
    processed_df = features

# 5) Visualize outliers and remove them
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[["age", "fare"]].melt(var_name="feature", value_name="value"), x="feature", y="value")
plt.title("Boxplot of numerical features before outlier removal")
plt.tight_layout()
plt.savefig("titanic_boxplot_before.png")
plt.close()

# Remove outliers using z-score on age and fare
for col in ["age", "fare"]:
    if col in df.columns:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z < 3]

print("\nShape after outlier removal:", df.shape)

# Re-run preprocessing after filtering
for col in ["age", "fare"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in ["embarked", "deck", "embark_town"]:
    if col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

if "survived" in encoded_df.columns:
    target = encoded_df["survived"]
    features = encoded_df.drop(columns=["survived"])
else:
    target = None
    features = encoded_df

numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    mean = features[col].mean()
    std = features[col].std()
    if std != 0:
        features[col] = (features[col] - mean) / std

if target is not None:
    processed_df = pd.concat([target, features], axis=1)
else:
    processed_df = features

print("\nProcessed data preview:")
print(processed_df.head())
print("\nProcessed shape:", processed_df.shape)
print("\nProcessed columns:")
print(processed_df.columns.tolist())

processed_df.to_csv("titanic_preprocessed.csv", index=False)
print("\nSaved cleaned dataset to titanic_preprocessed.csv")
