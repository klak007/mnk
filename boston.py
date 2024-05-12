# Importing Libraries and Reading the Dataset
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from mnk_functions import load_and_preprocess_boston_data, visualize_data, run_model, create_model_summary, \
    plot_model_performance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_style('darkgrid')

file_path = "boston.txt"
X, y, raw_df = load_and_preprocess_boston_data(file_path)

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print shapes of the training and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Convert the features and target back to a DataFrame for visualization
dataset = pd.DataFrame(X, columns=raw_df.columns[:-1])
dataset['MEDV'] = y

# visualize_data(dataset)  # Visualize the data

rmse_linear, r2_score_linear_train, r2_score_linear_test, cv_linear = run_model(LinearRegression(), X_train, y_train,
                                                                                X_test, y_test, "Linear Regression")

# Define the pipeline
steps = [
    # ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3.8, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)

# Ridge Regression
rmse_ridge, r2_score_ridge_train, r2_score_ridge_test, cv_ridge = run_model(ridge_pipe,
                                                                            X_train, y_train, X_test, y_test,
                                                                            "Ridge Regression")
steps = [
    # ('sc_X', StandardScaler()),
    ('model', SVR(kernel='rbf', gamma='scale'))
]
svr_pipe = Pipeline(steps)

# Support Vector Regression
rmse_svr, r2_score_svr_train, r2_score_svr_test, cv_svr = run_model(svr_pipe,
                                                                    X_train, y_train.ravel(), X_test, y_test.ravel(),
                                                                    "Support Vector Regression")

# Create model summaries
linear_summary = create_model_summary("Linear Regression", rmse_linear, r2_score_linear_train, r2_score_linear_test,
                                      cv_linear)
ridge_summary = create_model_summary("Ridge Regression", rmse_ridge, r2_score_ridge_train, r2_score_ridge_test,
                                     cv_ridge)
svr_summary = create_model_summary("Support Vector Regression", rmse_svr, r2_score_svr_train, r2_score_svr_test, cv_svr)

# Concatenate all the summaries into one DataFrame
model_summaries = pd.concat([linear_summary, ridge_summary, svr_summary], ignore_index=True)

plot_model_performance(model_summaries)

# save the results to a CSV file with the one more column for the dataset name
model_summaries['Dataset'] = 'Boston Housing'
model_summaries.to_csv('results.csv', index=False)


