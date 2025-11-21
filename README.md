# Diabetes_Prediction
🩺 Pima Indian Diabetes Prediction Model
# COLAB LINK : https://colab.research.google.com/drive/1Ruif6ZlUCPw0hrH8c4HhNEKYXA6YXUwr?usp=sharing
import pandas as pd

# Load the dataset from 'diabetes.csv' into a pandas DataFrame
df = pd.read_csv('diabetes.csv')

# Display the first 5 rows of the DataFrame
print("First 5 rows of the DataFrame:")
print(df.head())

# Display a concise summary of the DataFrame
print("\nDataFrame Information:")
df.info()
import numpy as np

# Columns where 0 values are likely missing data
columns_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 values with NaN in the identified columns
df[columns_with_zero_as_missing] = df[columns_with_zero_as_missing].replace(0, np.nan)

# Impute missing values (NaNs) with the mean of each column
for column in columns_with_zero_as_missing:
    df[column].fillna(df[column].mean(), inplace=True)

print("Missing values handled and imputed.")
print("First 5 rows of DataFrame after imputation:")
print(df.head())
print("\nDataFrame Information after imputation:")
df.info()
from sklearn.preprocessing import StandardScaler

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to a DataFrame with original column names
X = pd.DataFrame(X_scaled, columns=X.columns)

print("Features and target separated and numerical features scaled.")
print("First 5 rows of scaled features (X):")
print(X.head())
print("First 5 rows of target variable (y):")
print(y.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate Logistic Regression model
model = LogisticRegression(random_state=42, solver='liblinear') # Added solver for FutureWarning

# Train the model
model.fit(X_train, y_train)

print("Data split into training and testing sets.")
print("Logistic Regression model trained successfully.")
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Diabetes Prediction')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
