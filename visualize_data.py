import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load the data
try:
    df = pd.read_csv('milk_readings.csv')
except FileNotFoundError:
    print("milk_readings.csv not found. Please ensure the file exists in the current directory.")
    sys.exit(1)

# Use actual column names from the CSV
required_columns = {'ph', 'turbidity', 'ec', 'protein', 'scc', 'milk_quality'}
if not required_columns.issubset(df.columns):
    print(f"CSV file must contain columns: {required_columns}")
    print(f"Found columns: {set(df.columns)}")
    sys.exit(1)

# Boxplots for each feature by Milk Quality
features = ['ph', 'turbidity', 'ec', 'protein', 'scc']
plt.figure(figsize=(18, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x='milk_quality', y=feature, palette='Set2')
    plt.title(f'{feature} by Milk Quality')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Distribution plots
features = ['ph', 'turbidity', 'ec', 'protein', 'scc']
plt.figure(figsize=(15, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show() 