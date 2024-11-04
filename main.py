# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# Check for missing values in the dataset
print("\nMissing values:\n", df.isnull().sum())



# Machine Learning: Decision Tree
# Create and train the Decision Tree model on the training data
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluation of the model's performance using ROC-AUC score and Matthews Correlation Coefficient
roc_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
mcc = matthews_corrcoef(y_test, y_pred_dt)

print("\nDecision Tree ROC-AUC Score:", roc_auc)
print("Decision Tree Matthews Correlation Coefficient (MCC):", mcc)




print(df)


