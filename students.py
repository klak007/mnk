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
# Load the students dataset from the text file
students = pd.read_csv(file_path, sep=',')
print(students.head())

# Split the dataset into the data and target
students_data = students.drop('Chance of Admit ', axis=1)  # Notice the space in 'Chance of Admit '
students_target = students['Chance of Admit ']

# Split the dataset into the Training set and Test set
students_train_data, students_test_data, students_train_target, students_test_target = train_test_split(students_data, students_target, test_size=0.2, random_state=25)

# Linear Regression
rmse, r2_train, r2_test, cv_mean = run_model(LinearRegression(), students_train_data, students_train_target, students_test_data, students_test_target, "Linear Regression")
linear_summary = create_model_summary("Linear Regression", rmse, r2_train, r2_test, cv_mean)

# Define the pipeline for Ridge Regression
ridge_steps = [
    ('scalar', StandardScaler()),
    ('model', Ridge(alpha=1.0))
]
ridge_pipe = Pipeline(ridge_steps)

# Ridge Regression
rmse, r2_train, r2_test, cv_mean = run_model(ridge_pipe, students_train_data, students_train_target, students_test_data, students_test_target, "Ridge Regression")
ridge_summary = create_model_summary("Ridge Regression", rmse, r2_train, r2_test, cv_mean)

# Define the pipeline for Support Vector Regression
svr_steps = [
    ('sc_X', StandardScaler()),
    ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
]
svr_pipe = Pipeline(svr_steps)

# Support Vector Regression
rmse, r2_train, r2_test, cv_mean = run_model(svr_pipe, students_train_data, students_train_target.values.ravel(), students_test_data, students_test_target.values.ravel(), "Support Vector Regression")
svr_summary = create_model_summary("Support Vector Regression", rmse, r2_train, r2_test, cv_mean)

# Combine the summaries and plot the model performance
model_summaries = pd.concat([linear_summary, ridge_summary, svr_summary])
plot_model_performance(model_summaries)
model_summaries['Dataset'] = 'Students Admission Data'
model_summaries.to_csv('results.csv', mode='a', header=False, index=False)