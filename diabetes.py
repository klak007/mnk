from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from mnk_functions import run_model, create_model_summary, plot_model_performance
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import sys
file_path = sys.argv[1]
# Load the diabetes dataset from the text file
diabetes = pd.read_csv(file_path, sep='\t')
print(diabetes.head())

# Split the dataset into the data and target
diabetes_data = diabetes.drop('Y', axis=1)
diabetes_target = diabetes['Y']

# Split the dataset into the Training set and Test set
patients_train_data, patients_test_data, patients_train_target, patients_test_target = train_test_split(diabetes_data, diabetes_target, test_size=0.2, random_state=25)

# Linear Regression
rmse, r2_train, r2_test, cv_mean = run_model(LinearRegression(), patients_train_data, patients_train_target, patients_test_data, patients_test_target, "Linear Regression")
linear_summary = create_model_summary("Linear Regression", rmse, r2_train, r2_test, cv_mean)

# Define the pipeline for Ridge Regression
ridge_steps = [
    ('scalar', StandardScaler()),
    ('model', Ridge(alpha=1.0))
]
ridge_pipe = Pipeline(ridge_steps)

# Ridge Regression
rmse, r2_train, r2_test, cv_mean = run_model(ridge_pipe, patients_train_data, patients_train_target, patients_test_data, patients_test_target, "Ridge Regression")
ridge_summary = create_model_summary("Ridge Regression", rmse, r2_train, r2_test, cv_mean)

# Define the pipeline for Support Vector Regression
svr_steps = [
    ('sc_X', StandardScaler()),
    ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
]
svr_pipe = Pipeline(svr_steps)

# Support Vector Regression
rmse, r2_train, r2_test, cv_mean = run_model(svr_pipe, patients_train_data, patients_train_target.ravel(), patients_test_data, patients_test_target.ravel(), "Support Vector Regression")
svr_summary = create_model_summary("Support Vector Regression", rmse, r2_train, r2_test, cv_mean)

# Combine the summaries and plot the model performance
model_summaries = pd.concat([linear_summary, ridge_summary, svr_summary])

# add the results at the end to a CSV file with the one more column for the dataset name, there can be data already in the results.csv file from other datasets

model_summaries['Dataset'] = 'Diabetes'
model_summaries.to_csv('results.csv', mode='a', header=False, index=False)


plot_model_performance(model_summaries)
