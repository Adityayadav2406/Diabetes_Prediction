# Diabetes_Prediction
🩺 Pima Indian Diabetes Prediction Model
# COLAB LINK : https://colab.research.google.com/drive/1Ruif6ZlUCPw0hrH8c4HhNEKYXA6YXUwr?usp=sharing
#Dataset File: •	https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Task
Build, evaluate, and visualize a machine learning model to predict diabetes using the `diabetes.csv` dataset.
## Load Data

### Subtask:
Load the diabetes dataset from the 'diabetes.csv' file into a pandas DataFrame.

**Reasoning**:
First, I'll import the pandas library to enable data manipulation. Then, I'll load the specified CSV file into a DataFrame, display its initial rows to verify the data was loaded correctly, and provide a summary of the DataFrame's structure and data types.


## Preprocess Data

### Subtask:
Clean and preprocess the data, handling missing values, encoding categorical features if any, and scaling numerical features as needed for model training.

**Reasoning**:
First, I will identify the columns where a value of '0' indicates a missing value and replace these '0's with `NaN` to prepare for imputation. Then, I will impute these `NaN` values with the mean of their respective columns to ensure data completeness.


**Reasoning**:
The previous step successfully handled missing values. Now, I will separate the features (X) from the target variable (y) and then scale the numerical features using `StandardScaler` to prepare the data for model training.


## Train Model

### Subtask:
Split the data into training and testing sets, then train a machine learning model (e.g., Logistic Regression, Random Forest, or SVM) to predict diabetes.

**Reasoning**:
I will import necessary libraries, split the data into training and testing sets, and then train a Logistic Regression model as per the instructions.


## Evaluate Model and Predict

### Subtask:
Evaluate the trained model's performance on the test set using appropriate metrics (e.g., accuracy, precision, recall, F1-score) and make predictions.

**Reasoning**:
I will make predictions on the test set, calculate the accuracy, and generate a classification report to evaluate the model's performance.


## Visualize Results

### Subtask:
Generate a visualization, such as a confusion matrix or ROC curve, to illustrate the model's performance. Ensure legends are provided.
**Reasoning**:
I will import the necessary libraries, compute the confusion matrix, and then visualize it as a heatmap to illustrate the model's performance as requested by the subtask.


## Final Task

### Subtask:
Summarize the model's performance, key findings from the prediction, and potential next steps for improving the model.


## Summary:

### Data Analysis Key Findings

*   The `diabetes.csv` dataset, containing 768 entries and 9 columns, was successfully loaded. Initial inspection revealed no explicitly missing values, although some columns contained '0's representing missing data.
*   Data preprocessing involved replacing '0' values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` columns with `NaN`, followed by imputation with the mean of their respective columns. All numerical features were then scaled using `StandardScaler`.
*   A Logistic Regression model was trained on the preprocessed data, achieving an accuracy of 0.7532 on the test set.
*   The model demonstrated better performance in predicting individuals without diabetes (Class 0), with a precision of 0.80, recall of 0.83, and F1-score of 0.81.
*   For predicting individuals with diabetes (Class 1), the model's performance was lower, showing a precision of 0.67, recall of 0.62, and F1-score of 0.64.
*   A confusion matrix visualization was generated, providing a clear illustration of the true positives, true negatives, false positives, and false negatives from the model's predictions.

### Insights or Next Steps

*   The current model performs reasonably well overall but has a noticeable disparity in performance between predicting non-diabetic and diabetic cases, with a lower recall for the diabetic class (0.62). This suggests the model is more prone to missing actual diabetes cases, which is a critical concern in a medical diagnostic context.
*   To improve the model's ability to identify diabetic patients, future steps could involve exploring advanced machine learning models (e.g., Gradient Boosting, Support Vector Machines), employing techniques to address potential class imbalance if present, or conducting feature engineering to derive more predictive variables.
# visualization image 
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/ba0725c9-3b58-48ca-b296-e481f9768b4a" />
