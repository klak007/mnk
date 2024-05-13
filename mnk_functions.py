import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import sys
file_path = sys.argv[1]

def load_and_preprocess_boston_data(file_path):
    # Define column names based on the Boston Housing dataset
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                    'MEDV']

    # Load the dataset from the provided file path, assuming it's space-separated
    dataset = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

    print(dataset.head())  # Display the first 5 rows of the dataset

    X = dataset.iloc[:, :-1].values  # Select all rows and all columns except the last one for features
    y = dataset.iloc[:, -1].values.reshape(-1,
                                           1)  # Select all rows and only the last column for the target, and reshape it

    return X, y, dataset


def visualize_data(dataset):
    # Calculate the correlation matrix
    corr = dataset.corr()

    # Create a new figure for the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate a heatmap
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f", ax=ax)

    # Apply xticks and yticks
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Show the plot
    # plt.show()

    # Create a pairplot
    # sns.pairplot(dataset)
    # plt.show()


def run_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)

    # Predicting Cross Validation Score the Test set results
    cv = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

    # Predicting R2 Score the Train set results
    y_pred_train = model.predict(X_train)
    r2_score_train = r2_score(y_train, y_pred_train)

    # Predicting R2 Score the Test set results
    y_pred_test = model.predict(X_test)
    r2_score_test = r2_score(y_test, y_pred_test)

    # Predicting RMSE the Test set results
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print(model_name)
    print("CV: ", cv.mean())
    print('R2_score (train): ', r2_score_train)
    print('R2_score (test): ', r2_score_test)
    print("RMSE: ", rmse)
    print("---------------------------")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=4)

    plt.xlabel('Actual ')
    plt.ylabel('Predicted ')
    plt.title('Actual vs Predicted ({})'.format(model_name))
    plt.legend()
    # save figure with the name of the model and the file name
    plt.savefig('plots/actual_vs_predicted_{}_{}.png'.format(model_name.lower().replace(' ', '_'), file_path.split('.')[0]))
    # plt.savefig('plots/actual_vs_predicted_{}.png'.format(model_name.lower().replace(' ', '_')))
    # plt.show()

    return rmse, r2_score_train, r2_score_test, cv.mean()


def create_model_summary(model_name, rmse, r2_train, r2_test, cv_mean):
    models = [(model_name, rmse, r2_train, r2_test, cv_mean)]
    model_summary = pd.DataFrame(data=models,
                                 columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)', 'Cross-Validation'])
    return model_summary


def plot_model_performance(model_summaries):
    # Set the aesthetic style of the plots
    sns.set_style('whitegrid')
    sns.set_context('talk')

    # Plot Cross-Validation Score
    f, axe = plt.subplots(1, 1, figsize=(20, 8))  # Increase the size of the figure
    model_summaries.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)
    sns.barplot(x='Cross-Validation', y='Model', data=model_summaries, ax=axe, palette='viridis')
    axe.set_xlabel('Cross-Validaton Score', size=20)
    axe.set_ylabel('Model', size=20)
    axe.set_xlim(0, 1.0)
    axe.set_yticklabels(axe.get_yticklabels(), rotation=30, ha='right')  # Rotate and align y-axis labels
    plt.title('Cross-Validation Score', size=24)
    plt.tight_layout()  # Adjust the layout
    plt.savefig('plots/cross_validation_summary_{}.png'.format(file_path.split('.')[0]))
    # plt.show()

    # Plot R2 Score (Training) and R2 Score (Test)
    f, axes = plt.subplots(2, 1, figsize=(14, 12))  # Increase the height of the figure
    model_summaries.sort_values(by=['R2_Score(training)'], ascending=False, inplace=True)
    sns.barplot(x='R2_Score(training)', y='Model', data=model_summaries, palette='viridis', ax=axes[0])
    axes[0].set_xlabel('R2 Score (Training)', size=20)
    axes[0].set_ylabel('Model', size=20)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title('R2 Score Training', size=24, pad=20)  # Add padding to the title

    model_summaries.sort_values(by=['R2_Score(test)'], ascending=False, inplace=True)
    sns.barplot(x='R2_Score(test)', y='Model', data=model_summaries, palette='viridis', ax=axes[1])
    axes[1].set_xlabel('R2 Score (Test)', size=20)
    axes[1].set_ylabel('Model', size=20)
    axes[1].set_xlim(0, 1.0)
    axes[1].set_title('R2 Score Test', size=24, pad=20)  # Add padding to the title
    plt.tight_layout()  # Adjust the layout
    plt.savefig('plots/r2_score_summary_{}.png'.format(file_path.split('.')[0]))
    # plt.show()

    # Plot RMSE
    model_summaries.sort_values(by=['RMSE'], ascending=False, inplace=True)
    f, axe = plt.subplots(1, 1, figsize=(18, 6))
    sns.barplot(x='Model', y='RMSE', data=model_summaries, ax=axe, palette='viridis')
    axe.set_xlabel('Model', size=20)
    axe.set_ylabel('RMSE', size=20)
    plt.title('RMSE', size=24)
    plt.savefig('plots/rmse_summary_{}.png'.format(file_path.split('.')[0]))
    # plt.show()
