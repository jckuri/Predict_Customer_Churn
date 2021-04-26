'''
FILE: churn_script_logging_and_tests.py
PURPOSE: This Python script runs all the tests that cover all the functions
         in the file churn_library.py.
AUTHOR: Juan Carlos Kuri Pinto
CREATION DATE: April 23, 2021.
'''

import unittest
import logging
import time

import churn_library as cl
import constants as C

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_function_with_dataframe(function_to_test, function_name):
    '''
    This method tests a function with a dataframe as parameter.

    Input:
            function_to_test: A function to test.
            function_name: The name of the function to test.
    Output:
            None.
    '''
    try:
        dataframe = cl.import_data(C.bank_data_file)
        time0 = time.time()
        function_to_test(dataframe)
        delta_time = time.time() - time0
        time_string = cl.seconds_to_time_string(delta_time)
        logging.info(
            'Testing %s: SUCCESS! Execution time: %s.',
            function_name,
            time_string)
    except Exception as err:
        logging.error(
            'Testing %s: The function %s produced an error.',
            function_name,
            function_name)
        raise err


def test_complex_function(function_to_test, function_name):
    '''
    This method tests a complex function.

    Input:
            function_to_test: A function to test.
            function_name: The name of the function to test.
    Output:
            None.
    '''
    try:
        time0 = time.time()
        function_to_test()
        delta_time = time.time() - time0
        time_string = cl.seconds_to_time_string(delta_time)
        logging.info(
            'Testing %s: SUCCESS! Execution time: %s.',
            function_name,
            time_string)
    except Exception as err:
        logging.error(
            'Testing %s: The function %s produced an error.',
            function_name,
            function_name)
        raise err

# pylint: disable=R0904


class TestChurnLibrary(unittest.TestCase):
    '''
    The TestChurnLibrary class contains all the tests to cover the
    functions in the Python script churn_library.py.
    '''

    def __init__(self, *args, **kwargs):
        '''
        This is the constructor of the class TestChurnLibrary.

        Input:
            args: List of arguments.
                kwargs: Dictionary of keyword arguments.
        Output:
            None.
        '''
        super().__init__(*args, **kwargs)

    # IMPORT DATA FUNCTION

    @classmethod
    def test_import_data(cls):
        '''
        This method tests the function import_data.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        try:
            dataframe = cl.import_data(C.bank_data_file)
            logging.info('Testing import_data: SUCCESS!')
        except FileNotFoundError as err:
            logging.error(
                'Testing import_data: The bank data file was not found.')
            raise err
        try:
            assert dataframe.shape[0] > 0
        except AssertionError as err:
            logging.error(
                'Testing import_data: The bank data file has no rows.')
            raise err
        try:
            assert dataframe.shape[1] > 0
        except AssertionError as err:
            logging.error(
                'Testing import_data: The bank data file has no columns.')
            raise err

    # EDA FUNCTIONS

    @classmethod
    def test_plot_churn_histogram(cls):
        '''
        This method tests the function plot_churn_histogram.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        test_function_with_dataframe(
            cl.plot_churn_histogram,
            'plot_churn_histogram')

    @classmethod
    def test_plot_customer_age_histogram(cls):
        '''
        This method tests the function plot_customer_age_histogram.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        test_function_with_dataframe(
            cl.plot_customer_age_histogram,
            'plot_customer_age_histogram')

    @classmethod
    def test_plot_marital_status_percentages(cls):
        '''
        This method tests the function plot_marital_status_percentages.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        test_function_with_dataframe(
            cl.plot_marital_status_percentages,
            'plot_marital_status_percentages')

    @classmethod
    def test_plot_correlation_heatmap(cls):
        '''
        This method tests the function plot_correlation_heatmap.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        test_function_with_dataframe(
            cl.plot_correlation_heatmap,
            'plot_correlation_heatmap')

    @classmethod
    def test_plot_transaction_count_distribution(cls):
        '''
        This method tests the function plot_transaction_count_distribution.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        test_function_with_dataframe(
            cl.plot_transaction_count_distribution,
            'plot_transaction_count_distribution')

    @classmethod
    def test_perform_eda(cls):
        '''
        This method tests the function perform_eda.
        '''
        test_function_with_dataframe(cl.perform_eda, 'perform_eda')

    # FEATURE ENGINEERING FUNCTIONS

    @classmethod
    def test_encode_categorical_feature(cls):
        '''
        This method tests the function encode_categorical_feature.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def encode_categorical_feature_function():
            dataframe = cl.import_data(C.bank_data_file)
            feature = 'Gender'
            cl.encode_categorical_feature(
                dataframe, feature, 'Churn', f'{feature}_Churn')
        test_complex_function(
            encode_categorical_feature_function,
            'encode_categorical_feature')

    @classmethod
    def test_encoder_helper(cls):
        '''
        This method tests the function encoder_helper.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def encoder_helper_function():
            dataframe = cl.import_data(C.bank_data_file)
            dataframe = cl.encoder_helper(dataframe,
                                          ['Gender',
                                           'Education_Level',
                                           'Marital_Status',
                                           'Income_Category',
                                           'Card_Category'])
        test_complex_function(encoder_helper_function, 'encoder_helper')

    @classmethod
    def test_perform_feature_engineering(cls):
        '''
        This method tests the function perform_feature_engineering.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        test_function_with_dataframe(
            cl.perform_feature_engineering,
            'perform_feature_engineering')

    # REPORT FUNCTIONS

    @classmethod
    def test_classification_report_image(cls):
        '''
        This method tests the function classification_report_image.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def classification_report_image_function():
            dataframe = cl.import_data(C.bank_data_file)
            x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
                dataframe)
            rfc, lrc = cl.load_models()
            y_train_preds_lr, _, y_test_preds_lr, _ = cl.compute_predictions(
                rfc, lrc, x_train, x_test)
            cl.classification_report_image(
                f'{C.results_images_path}/lr_classification_report.png',
                'Logistic Regression',
                (y_train, y_test, y_train_preds_lr, y_test_preds_lr))
        test_complex_function(
            classification_report_image_function,
            'classification_report_image')

    @classmethod
    def test_classification_report_images(cls):
        '''
        This method tests the function classification_report_images.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def classification_report_images_function():
            dataframe = cl.import_data(C.bank_data_file)
            x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
                dataframe)
            rfc, lrc = cl.load_models()
            y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = \
                cl.compute_predictions(rfc, lrc, x_train, x_test)
            cl.classification_report_images(
                (y_train, y_test),
                (y_train_preds_lr, y_test_preds_lr),
                (y_train_preds_rf, y_test_preds_rf))
        test_complex_function(
            classification_report_images_function,
            'classification_report_images')

    # FEATURE IMPORTANCE FUNCTIONS

    @classmethod
    def test_plot_shap_feature_importances(cls):
        '''
        This method tests the function plot_shap_feature_importances.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def plot_shap_feature_importances_function():
            dataframe = cl.import_data(C.bank_data_file)
            _, x_test, _, _ = cl.perform_feature_engineering(
                dataframe)
            rfc, _ = cl.load_models()
            cl.plot_shap_feature_importances(
                rfc, x_test, f'{C.results_images_path}/rf_shap_feature_importances.png')
        test_complex_function(
            plot_shap_feature_importances_function,
            'plot_shap_feature_importances')

    @classmethod
    def test_plot_feature_importances(cls):
        '''
        This method tests the function plot_feature_importances.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def plot_feature_importances_function():
            dataframe = cl.import_data(C.bank_data_file)
            _, x_test, _, _ = cl.perform_feature_engineering(
                dataframe)
            rfc, _ = cl.load_models()
            cl.plot_feature_importances(
                rfc, x_test, f'{C.results_images_path}/rf_feature_importances.png')
        test_complex_function(
            plot_feature_importances_function,
            'plot_feature_importances')

    @classmethod
    def test_feature_importance_plots(cls):
        '''
        This method tests the function feature_importance_plots.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def feature_importance_plots_function():
            dataframe = cl.import_data(C.bank_data_file)
            _, x_test, _, _ = cl.perform_feature_engineering(
                dataframe)
            rfc, _ = cl.load_models()
            cl.feature_importance_plots(rfc, x_test)
        test_complex_function(
            feature_importance_plots_function,
            'feature_importance_plots')

    # ROC FUNCTIONS

    @classmethod
    def test_draw_roc_curve(cls):
        '''
        This method tests the function draw_roc_curve.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def draw_roc_curve_function():
            dataframe = cl.import_data(C.bank_data_file)
            _, x_test, _, y_test = cl.perform_feature_engineering(
                dataframe)
            _, lrc = cl.load_models()
            cl.draw_roc_curve(
                lrc,
                x_test,
                y_test,
                f'{C.results_images_path}/lr_roc_curve.png')
        test_complex_function(draw_roc_curve_function, 'draw_roc_curve')

    @classmethod
    def test_draw_both_roc_curves(cls):
        '''
        This method tests the function draw_both_roc_curves.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def draw_both_roc_curves_function():
            dataframe = cl.import_data(C.bank_data_file)
            _, x_test, _, y_test = cl.perform_feature_engineering(
                dataframe)
            rfc, lrc = cl.load_models()
            cl.draw_both_roc_curves(
                rfc,
                lrc,
                x_test,
                y_test,
                f'{C.results_images_path}/roc_curves.png')
        test_complex_function(
            draw_both_roc_curves_function,
            'draw_both_roc_curves')

    @classmethod
    def test_draw_all_roc_curves(cls):
        '''
        This method tests the function draw_all_roc_curves.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def draw_all_roc_curves_function():
            dataframe = cl.import_data(C.bank_data_file)
            _, x_test, _, y_test = cl.perform_feature_engineering(
                dataframe)
            rfc, lrc = cl.load_models()
            cl.draw_all_roc_curves(rfc, lrc, x_test, y_test)
        test_complex_function(
            draw_all_roc_curves_function,
            'draw_all_roc_curves')

    # MODEL FUNCTIONS

    @classmethod
    def test_load_models_and_save_models(cls):
        '''
        This method tests the functions load_models and save_models.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def load_models_and_save_models_function():
            rfc, lrc = cl.load_models()
            cl.save_models(rfc, lrc)
        test_complex_function(
            load_models_and_save_models_function,
            'load_models_and_save_models')

    # TRAIN FUNCTION

    @classmethod
    def test_train_models(cls):
        '''
        This method tests the function train_models.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def train_models_function():
            dataframe = cl.import_data(C.bank_data_file)
            x_train, _, y_train, _ = cl.perform_feature_engineering(
                dataframe)
            cl.train_models(x_train, y_train)
        test_complex_function(train_models_function, 'train_models')

    # HELPER FUNCTIONS

    @classmethod
    def test_compute_predictions(cls):
        '''
        This method tests the function compute_predictions.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        def compute_predictions_function():
            dataframe = cl.import_data(C.bank_data_file)
            x_train, x_test, _, _ = cl.perform_feature_engineering(
                dataframe)
            rfc, lrc = cl.load_models()
            _, _, _, _ = cl.compute_predictions(
                rfc, lrc, x_train, x_test)
        test_complex_function(
            compute_predictions_function,
            'compute_predictions')

    @classmethod
    def test_seconds_to_time_string(cls):
        '''
        This method tests the function seconds_to_time_string.

        Input:
            cls: Class of the method.
        Output:
            None.
        '''
        try:
            assert(cl.seconds_to_time_string(122.22)
                   == '2 minutes 2.22 seconds')
            assert(cl.seconds_to_time_string(1202.22)
                   == '20 minutes 2.22 seconds')
            assert(cl.seconds_to_time_string(362.0)
                   == '6 minutes 2.00 seconds')
            logging.info('Testing seconds_to_time_string: SUCCESS!')
        except Exception as err:
            logging.error(
                'Testing seconds_to_time_string: ERROR!')
            raise err


if __name__ == "__main__":
    unittest.main()
