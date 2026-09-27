import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())
print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

print("\nBasic Stats (Amount & Time):")
print(df[["Time", "Amount"]].describe())

# Compare average amount between normal and fraud transactions
print("Average amount - Normal transactions:", round(df[df["Class"]==0]["Amount"].mean(), 2))
print("Average amount - Fraud transactions:", round(df[df["Class"]==1]["Amount"].mean(), 2))

# Convert time to hour of day
df["Hour"] = (df["Time"] // 3600) % 24

# Top columns correlated with fraud
correlations = df.corr()["Class"].sort_values()
print("\nTop columns correlated with fraud:")
print(correlations.head(5))
print(correlations.tail(5))

# 1. Plot Class Distribution (Normal vs Fraud)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (Log Scale)')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')
plt.yscale('log')  # Log scale due to severe class imbalance
plt.savefig('class_distribution.png')
plt.show()

# 2. Plot Top Features Correlated with Fraud
plt.figure(figsize=(8, 5))
top_corrs = pd.concat([correlations.head(5), correlations.tail(5)])
top_corrs.plot(kind='barh', color='teal')
plt.title('Top Correlations with Fraud')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('top_correlations.png')
plt.show()

# 3. Fraud Transactions Distribution by Hour of Day
plt.figure(figsize=(10, 4))
sns.histplot(data=df[df['Class'] == 1], x='Hour', bins=24, color='red', kde=True)
plt.title('Fraud Transactions by Hour of Day')
plt.xlabel('Hour (0 - 23)')
plt.ylabel('Fraud Count')
plt.tight_layout()
plt.savefig('fraud_by_hour.png')
plt.show()
