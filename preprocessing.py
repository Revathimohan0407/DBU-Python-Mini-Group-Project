# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the dataset
df = pd.read_csv('youth_smoking_drug_data_10000_rows_expanded.csv')
print("Dataset loaded successfully.\n")

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




