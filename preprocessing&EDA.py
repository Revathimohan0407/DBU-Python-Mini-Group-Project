# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Load the dataset
# Make sure the dataset is in the same directory or provide the full path to the dataset
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('youth_smoking_drug_data_10000_rows_expanded.csv')

# Data Exploration
# Display the first few rows of the dataset to get an overview
#print("First 5 rows of the dataset:\n", df.head())

# Display summary statistics for numerical columns
#print("\nSummary statistics:\n", df.describe())

# Check for missing values in the dataset
#print("\nMissing values:\n", df.isnull().sum())

# Print the dataframe to verify loading
print(df)

# Data Preprocessing

# Encoding categorical variables with text values before scaling
label_encoder = LabelEncoder()

# Apply label encoding to all categorical columns with non-numeric data
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Age_Group'] = label_encoder.fit_transform(df['Age_Group'])
df['Socioeconomic_Status'] = label_encoder.fit_transform(df['Socioeconomic_Status'])  # Encoding values like 'High', 'Low'

# Scale only the numerical columns
scaler = StandardScaler()
df[['Smoking_Prevalence', 'Drug_Experimentation', 'Socioeconomic_Status']] = scaler.fit_transform(
    df[['Smoking_Prevalence', 'Drug_Experimentation', 'Socioeconomic_Status']]
)

# Print first few rows to check transformations
print(df.head())

# Data Visualization (from the previous code)

# Box Plot for Smoking Prevalence by Age Group
sns.boxplot(x='Age_Group', y='Smoking_Prevalence', data=df)
plt.title("Smoking Prevalence by Age Group")
plt.show()

# Convert 'Yes'/'No' columns to 1/0
df = df.replace({'Yes': 1, 'No': 0})

# Check if all columns are numeric
print(df.dtypes)

# Heatmap for Correlation
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot for Peer Influence vs Drug Experimentation by Gender
sns.scatterplot(x='Peer_Influence', y='Drug_Experimentation', hue='Gender', data=df)
plt.title("Peer Influence vs Drug Experimentation")
plt.show()

# Proceed with model training or further analysis after this section


