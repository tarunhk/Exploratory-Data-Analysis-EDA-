import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("titanic.csv")

print("First rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# histogram
df.hist(figsize=(10,8))
plt.show()

# boxplot
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

# correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# pairplot
sns.pairplot(df[['Survived','Age','Fare','Pclass']])
plt.show()
