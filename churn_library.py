'''
FILE: churn_library.py
PURPOSE: This Python script contains the whole pipeline for the project
         Predict Customer Churn:
         - Loading the bank data file.
         - Performing the Exploratory Data Analysis (EDA).
         - Performing feature engineering.
         - Training models
         - Doing grid search over many random forest models.
         - Saving models.
         - Doing classification reports.
         - Drawing feature importance plots.
         - Drawing ROC curves.
AUTHOR: Juan Carlos Kuri Pinto
CREATION DATE: April 23, 2021.
'''

# IMPORT LIBRARIES

import os
import time
import sklearn
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as C

sns.set()

# IMPORT DATA FUNCTION


def import_data(csv_file):
    '''
    Reads a CSV file and returns its dataframe.

    Input:
        csv_file: A path to the CSV file.
    Output:
        dataframe: The Pandas dataframe with the bank data.
    '''
    dataframe = pd.read_csv(csv_file)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    return dataframe

# EDA FUNCTIONS


def plot_churn_histogram(dataframe):
    '''
    Plots the churn distribution.

    Input:
        dataframe: The Pandas dataframe with the bank data.
    Output:
        None.
    '''
    fig, axis = plt.subplots(constrained_layout=True)
    churn = dataframe.Churn.value_counts()
    axis.bar(['Not Churn', 'Churn'], churn)
    plt.xlabel('Churn')
    plt.ylabel('Customer Count')
    plt.title('Churn Counts')
    fig.savefig(f'{C.eda_images_path}/churn_distribution.png')


def plot_customer_age_histogram(dataframe):
    '''
    Plots the customer age distribution.

    Input:
        dataframe: The Pandas dataframe with the bank data.
    Output:
        None.
    '''
    fig, _ = plt.subplots(constrained_layout=True)
    customer_age = dataframe['Customer_Age']
    plt.hist(customer_age)
    plt.xlabel('Customer Age')
    plt.ylabel('Customer Count')
    plt.title('Customer Age Histogram')
    fig.savefig(f'{C.eda_images_path}/customer_age_distribution.png')


def plot_marital_status_percentages(dataframe):
    '''
    Plots the marital status distribution.

    Input:
        dataframe: The Pandas dataframe with the bank data.
    Output:
        None.
    '''
    fig, axis = plt.subplots(constrained_layout=True)
    dataframe.Marital_Status.value_counts(
        'normalize').plot(kind='bar', ax=axis)
    plt.xlabel('Marital Status')
    plt.ylabel('Percentages')
    plt.title('Plot of Marital Status')
    fig.savefig(f'{C.eda_images_path}/marital_status_distribution.png')


def plot_correlation_heatmap(dataframe):
    '''
    Plots the correlation heatmap between all variables in the dataframe.

    Input:
        dataframe: The Pandas dataframe with the bank data.
    Output:
        None.
    '''
    # https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html
    fig, axis = plt.subplots(constrained_layout=True)
    sns.heatmap(
        dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2,
        ax=axis)
    plt.title('Correlation Heatmap')
    fig.savefig(f'{C.eda_images_path}/correlation_heatmap.png')


def plot_transaction_count_distribution(dataframe):
    '''
    Plots the total transaction count distribution.

    Input:
        dataframe: The Pandas dataframe with the bank data.
    Output:
        None.
    '''
    fig, _ = plt.subplots(constrained_layout=True)
    sns.histplot(dataframe, x='Total_Trans_Ct', stat='density', kde=True)
    plt.xlabel('Total Transaction Count')
    plt.title('Histogram of Total Transaction Counts')
    fig.savefig(f'{C.eda_images_path}/total_transaction_distribution.png')


def perform_eda(dataframe):
    '''
    Performs an Exploratory Data Analysis (EDA) and saves
    figures to the images folder.

    Input:
        dataframe: The Pandas dataframe with the bank data.
    Output:
        None.
    '''
    plot_churn_histogram(dataframe)
    plot_customer_age_histogram(dataframe)
    plot_marital_status_percentages(dataframe)
    plot_transaction_count_distribution(dataframe)
    plot_correlation_heatmap(dataframe)

# FEATURE ENGINEERING FUNCTIONS


def encode_categorical_feature(
        dataframe,
        feature_name,
        y_name,
        new_feature_name):
    '''
    Helper function to turn a categorical column into a new column with
    its proportion of churn.

    Input:
        dataframe: The Pandas dataframe with the bank data.
        feature_name: The name of the categorical column.
        y_name: The name of the Churn column.
        new_feature_name: feature_name + '_Churn'
    Output:
        None.
    '''
    feature_groups = dataframe.groupby(feature_name).mean()[y_name]
    dataframe[new_feature_name] = [feature_groups.loc[val]
                                   for val in dataframe[feature_name]]


def encoder_helper(dataframe, category_list):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category.

    Input:
            dataframe: The Pandas dataframe with the bank data.
            category_list: List of columns that contain categorical features.
    Output:
            dataframe: The Pandas dataframe with the new columns.
    '''
    for feature in category_list:
        encode_categorical_feature(
            dataframe, feature, 'Churn', f'{feature}_Churn')
    return dataframe


def perform_feature_engineering(dataframe):
    '''
    Performs feature engineering on the dataframe with the bank data.

    Input:
              dataframe: The Pandas dataframe with the bank data.
    Output:
              x_train: X training data.
              x_test: X testing data.
              y_train: Y training data.
              y_test: Y testing data.
    '''
    class_y = dataframe['Churn']
    features_x = pd.DataFrame()
    dataframe = encoder_helper(dataframe,
                               ['Gender',
                                'Education_Level',
                                'Marital_Status',
                                'Income_Category',
                                'Card_Category'])
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    features_x[keep_cols] = dataframe[keep_cols]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features_x, class_y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

# REPORT FUNCTIONS


def classification_report_image(
        image_file,
        classifier_name,
        all_y):
    '''
    Draws the classification report for the results of a classifier.

    Input:
        image_file: An image file.
        classifier_name: The name of the classifier.
        all_y: A tuple containing y_train, y_test, y_train_preds, y_test_preds
    Output:
        None.
    '''
    y_train, y_test, y_train_preds, y_test_preds = all_y
    fig, axis = plt.subplots(constrained_layout=True)
    text = f'Classification Report for {classifier_name}\n\n'
    text += f'{classifier_name} Train\n\n'
    text += f'{sklearn.metrics.classification_report(y_train, y_train_preds)}\n\n'
    text += f'{classifier_name} Test\n\n'
    text += f'{sklearn.metrics.classification_report(y_test, y_test_preds)}'
    axis.text(0.5,
              0.5,
              text,
              {'fontsize': 10},
              fontproperties='monospace',
              horizontalalignment='center',
              verticalalignment='center')
    axis.axis('off')
    fig.savefig(image_file)


def classification_report_images(ground_y, lr_y, rf_y):
    '''
    Produces classification report for training and testing results and stores
    report as image in images folder.

    Input:
        ground_y: A tuple containing y_train, y_test.
        lr_y: A tuple containing y_train_preds_lr, y_test_preds_lr.
              The predictions done by the logistic regression classifier.
        rf_y: A tuple containing y_train_preds_rf, y_test_preds_rf.
              The predictions done by the random forests classifier.
    Output:
        None.
    '''
    y_train, y_test = ground_y
    y_train_preds_lr, y_test_preds_lr = lr_y
    y_train_preds_rf, y_test_preds_rf = rf_y
    classification_report_image(
        f'{C.results_images_path}/rf_classification_report.png',
        'Random Forest',
        (y_train, y_test, y_train_preds_rf, y_test_preds_rf))
    classification_report_image(
        f'{C.results_images_path}/lr_classification_report.png',
        'Logistic Regression',
        (y_train, y_test, y_train_preds_lr, y_test_preds_lr))

# FEATURE IMPORTANCE FUNCTIONS


def plot_shap_feature_importances(model, x_data, output_path):
    '''
    Plots the shap feature importances and stores the image in the images folder.

    Input:
        model: Model object containing feature_importances_.
        x_data: Pandas dataframe of X values.
        output_path: Path to store the figure.
    Output:
        None.
    '''
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/demo_constrained_layout.html
    plt.figure(constrained_layout=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    # https://github.com/slundberg/shap/issues/153
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.savefig(output_path)


def plot_feature_importances(model, x_data, output_path):
    '''
    Plots the feature importances and stores the image in the images folder.

    Input:
        model: Model object containing feature_importances_.
        x_data: Pandas dataframe of X values.
        output_path: Path to store the figure.
    Output:
        None.
    '''
    plt.figure(constrained_layout=True)
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_path)


def feature_importance_plots(rfc, x_data):
    '''
    Plots feature importances and stores the images in the images folder.

    Input:
        model: Model object containing feature_importances_.
        x_data: Pandas dataframe of X values.
    Output:
        None.
    '''
    plot_shap_feature_importances(
        rfc, x_data, f'{C.results_images_path}/rf_shap_feature_importances.png')
    plot_feature_importances(
        rfc, x_data, f'{C.results_images_path}/rf_feature_importances.png')

# ROC FUNCTIONS


def draw_both_roc_curves(rfc, lrc, x_test, y_test, output_path):
    '''
    Plots ROC curves and stores the image in the images folder.

    Input:
        rfc: Random Forests Classifier.
        lrc: Linear Regression Classifier.
        x_test: Pandas dataframe of X test values.
        y_test: Pandas dataframe of Y test values.
        output_path: The images folder.
    Output:
        None.
    '''
    _, axis = plt.subplots(constrained_layout=True)
    axis = plt.gca()
    sklearn.metrics.plot_roc_curve(rfc, x_test, y_test, ax=axis, alpha=0.8)
    sklearn.metrics.plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(output_path)


def draw_roc_curve(model, x_test, y_test, output_path):
    '''
    Plots ROC curve and stores the image in the images folder.

    Input:
        model: A classification model.
        x_test: Pandas dataframe of X test values.
        y_test: Pandas dataframe of Y test values.
        output_path: The images folder.
    Output:
        None.
    '''
    _, axis = plt.subplots(constrained_layout=True)
    sklearn.metrics.plot_roc_curve(model, x_test, y_test, ax=axis)
    plt.savefig(output_path)


def draw_all_roc_curves(rfc, lrc, x_test, y_test):
    '''
    Plots ROC curves and stores the images in the images folder.

    Input:
        rfc: Random Forests Classifier.
        lrc: Linear Regression Classifier.
        x_test: Pandas dataframe of X test values.
        y_test: Pandas dataframe of Y test values.
    Output:
        None.
    '''
    draw_roc_curve(
        lrc,
        x_test,
        y_test,
        f'{C.results_images_path}/lr_roc_curve.png')
    draw_roc_curve(
        rfc,
        x_test,
        y_test,
        f'{C.results_images_path}/rf_roc_curve.png')
    draw_both_roc_curves(
        rfc,
        lrc,
        x_test,
        y_test,
        f'{C.results_images_path}/roc_curves.png')

# MODEL FUNCTIONS


def save_models(rfc, lrc):
    '''
    Saves the Random Forests model and the Linear Regression model.

    Input:
        rfc: Random Forests Classifier.
        lrc: Linear Regression Classifier.
    Output:
        None.
    '''
    joblib.dump(rfc, f'{C.models_path}/rfc_model.pkl')
    joblib.dump(lrc, f'{C.models_path}/logistic_model.pkl')


def load_models():
    '''
    Loads the Random Forests model and the Linear Regression model.

    Input:
        None.
    Output:
        rfc: Random Forests Classifier.
        lrc: Linear Regression Classifier.
    '''
    rfc = joblib.load(f'{C.models_path}/rfc_model.pkl')
    lrc = joblib.load(f'{C.models_path}/logistic_model.pkl')
    return rfc, lrc

# TRAIN FUNCTION


def train_models(x_train, y_train):
    '''
    Trains and stores the models.

    Input:
        x_train: X training data
        y_train: y training data
    Output:
        None.
    '''
    rfc = sklearn.ensemble.RandomForestClassifier(random_state=42)
    lrc = sklearn.linear_model.LogisticRegression(
        solver='lbfgs', max_iter=1000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto'],  # 'sqrt'
        'max_depth': [5, 25, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = sklearn.model_selection.GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, verbose=4)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)
    save_models(cv_rfc.best_estimator_, lrc)

# HELPER FUNCTIONS


def compute_predictions(rfc, lrc, x_train, x_test):
    '''
    Helper function to compute predictions of 2 models given the X values.

    Input:
        rfc: Random Forests Classifier.
        lrc: Linear Regression Classifier.
        x_train: X training data.
        x_test: X testing data.
    Output:
        y_train_preds_lr: Predictions done by Linear Regression with
                          training data.
        y_train_preds_rf: Predictions done by Random Forests with
                          training data.
        y_test_preds_lr: Predictions done by Linear Regression with
                         testing data.
        y_test_preds_rf: Predictions done by Random Forests with
                         testing data.
    '''
    y_train_preds_rf = rfc.predict(x_train)
    y_test_preds_rf = rfc.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def seconds_to_time_string(seconds):
    '''
    Helper function to transform seconds into a string representation of
    minutes and seconds.

    Input:
        seconds: The number of seconds.
    Output:
        The string representation of minutes and seconds.
    '''
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    if minutes == 0:
        return f'{secs:.2f} seconds'
    return f'{minutes} minutes {secs:.2f} seconds'


def remove_all_images():
    '''
    Helper function to remove all images from the images folder.

    Input:
        None.
    Output:
        None.
    '''
    os.system(f'rm {C.eda_images_path}/*.*')
    os.system(f'rm {C.results_images_path}/*.*')

# MAIN FUNCTION


def main():
    '''
    Main function that executes the whole pipeline for the project
    Predict Customer Churn.

    Input:
        None.
    Output:
        None.
    '''
    time0 = time.time()
    remove_all_images()
    dataframe = import_data(C.bank_data_file)
    perform_eda(dataframe)
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe)
    train_models(x_train, y_train)
    rfc, lrc = load_models()
    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = compute_predictions(
        rfc, lrc, x_train, x_test)
    classification_report_images(
        (y_train, y_test),
        (y_train_preds_lr, y_test_preds_lr),
        (y_train_preds_rf, y_test_preds_rf))
    feature_importance_plots(rfc, x_test)
    draw_all_roc_curves(rfc, lrc, x_test, y_test)
    delta_time = time.time() - time0
    print(
        f'SUCCESS! Total Execution time: {seconds_to_time_string(delta_time)}.')


if __name__ == "__main__":
    main()
