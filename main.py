# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import seaborn as sns
import os

# Load the dataset
# Make sure the dataset is in the same directory or provide the full path to the dataset
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('youth_smoking_drug_data_10000_rows_expanded.csv')

# Data Exploration
# Display the first few rows of the dataset to get an overview
print("First 5 rows of the dataset:\n", df.head())

# Display summary statistics for numerical columns
print("\nSummary statistics:\n", df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Convert categorical columns using one-hot encoding
# Specify actual categorical columns in the dataset
categorical_columns = ['Age_Group', 'Gender', 'Socioeconomic_Status', 'Peer_Influence', 'School_Programs', 
                       'Family_Background', 'Mental_Health', 'Access_to_Counseling', 'Parental_Supervision',
                       'Substance_Education', 'Community_Support', 'Media_Influence']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print(df)

# Define features (X) and target (y)
# Assuming 'Smoking_Prevalence' is the target variable; adjust as necessary
X = df.drop(['Smoking_Prevalence'], axis=1)  # Features
y = df['Smoking_Prevalence']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Normalization (scaling between 0 and 1)
min_max_scaler = MinMaxScaler()
X_train_normalized = min_max_scaler.fit_transform(X_train)
X_test_normalized = min_max_scaler.transform(X_test)

# Optional: Print to verify transformations
print("\nStandardized Training Data (first 5 rows):")
print(pd.DataFrame(X_train_standardized, columns=X.columns).head())
print("\nNormalized Training Data (first 5 rows):")
print(pd.DataFrame(X_train_normalized, columns=X.columns).head())



# Machine Learning: Decision Tree
# Create and train the Decision Tree model on the training data
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_standardized, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluation of the model's performance using ROC-AUC score and Matthews Correlation Coefficient
roc_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
mcc = matthews_corrcoef(y_test, y_pred_dt)

print("\nDecision Tree ROC-AUC Score:", roc_auc)
print("Decision Tree Matthews Correlation Coefficient (MCC):", mcc)

