# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, classification_report, confusion_matrix
import os
import seaborn as sns

# Load the dataset
# Make sure the dataset is in the same directory or provide the full path to the dataset
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('youth_smoking_drug_data_10000_rows_expanded.csv')

# Data Exploration
print("First 5 rows of the dataset:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("Missing values per column:")
print(df.isnull().sum())

# Visualization 1: Boxplot for Smoking Prevalence by Age Group
plt.figure(figsize=(8, 6))
sns.boxplot(x='Age_Group', y='Smoking_Prevalence', data=df)
plt.title("Smoking Prevalence by Age Group")
plt.show()
plt.clf()
 
# Visualization 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
plt.clf()
 
# Visualization 3: Scatter Plot for Peer Influence vs Drug Experimentation by Gender
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Peer_Influence', y='Drug_Experimentation', hue='Gender', data=df)
plt.title("Peer Influence vs Drug Experimentation")
plt.show()
plt.clf()

# Convert categorical columns using one-hot encoding
categorical_columns = ['Age_Group', 'Gender', 'Socioeconomic_Status', 'Peer_Influence', 'School_Programs', 
                       'Family_Background', 'Mental_Health', 'Access_to_Counseling', 'Parental_Supervision',
                       'Substance_Education', 'Community_Support', 'Media_Influence']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print(df)

# Define features (X) and target (y)
X = df.drop(['Smoking_Prevalence'], axis=1)  # Features
# Binarize `Smoking_Prevalence` target based on mean threshold
threshold = df['Smoking_Prevalence'].mean()
df['Smoking_Prevalence'] = (df['Smoking_Prevalence'] > threshold).astype(int)
y = df['Smoking_Prevalence']  # Binary Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Print to verify transformations
print("\nStandardized Training Data (first 5 rows):")
print(pd.DataFrame(X_train_standardized, columns=X.columns).head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
# Load the dataset
df = pd.read_csv('/Users/remo/Downloads/youth_smoking_drug_data_10000_rows_expanded.csv')
 
# Encode categorical columns
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Age_Group'] = label_encoder.fit_transform(df['Age_Group'])
df['Socioeconomic_Status'] = label_encoder.fit_transform(df['Socioeconomic_Status'])
 
# Scale numerical columns
scaler = StandardScaler()
df[['Smoking_Prevalence', 'Drug_Experimentation', 'Socioeconomic_Status']] = scaler.fit_transform(
    df[['Smoking_Prevalence', 'Drug_Experimentation', 'Socioeconomic_Status']]
)
# Machine Learning: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_standardized, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test_standardized)

# Evaluation of the model's performance
roc_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test_standardized)[:, 1])
mcc = matthews_corrcoef(y_test, y_pred_dt)

print("\nDecision Tree ROC-AUC Score:", roc_auc)
print("Decision Tree Matthews Correlation Coefficient (MCC):", mcc)
