"""
This is for model implementation

MCCT (initial)/ Minnie Cherry Chua Tan 9-Jul-21 Base version coded upgrade to version 2
    Added dict_to_str_keyvalue()
MCCT (initial)/ Minnie Cherry Chua Tan 16-Jul-21 Added dict2df()
MCCT (initial)/ Minnie Cherry Chua Tan 24-Jul-21 Added method_param_transform(), changed
    dict2df()
MCCT (initial)/ Minnie Cherry Chua Tan 24-Jul-21 change dict2df() for dataframe of key-value
    pairs and method calls
MCCT (initial)/ Minnie Cherry Chua Tan 22-Aug-21, 23-Aug-21 Update init() for model and feature reduction variables
MCCT (initial)/ Minnie Cherry Chua Tan 24-Aug-21 Update init() for scores statistics handling, added summary_stats()
MCCT (initial)/ Minnie Cherry Chua Tan 01-Sep-21 Updated summary_stats() to spool in excel and separate the handling of regrssor and classifier
MCCT (initial)/ Minnie Cherry Chua Tan 02-Sep-21 Update to handle predict_test in the input file, update init() for SMOOTH graphing
MCCT (initial)/ Minnie Cherry Chua Tan 03-Sep-21 Update to refactor the stats_df to be global
MCCT (initial)/ Minnie Cherry Chua Tan 07-Sep-21 Update SMOOTH for classification as a list
MCCT (initial)/ Minnie Cherry Chua Tan 13-Sep-21 Added xlabel_reduced for the reduced label
MCCT (initial)/ Minnie Cherry Chua Tan 20-Sep-21 Added cluster scores in summary_stats(), init(),
    refactor from regression_classification.py - set_label(), plot_report(), plot_pairplot(), added
    report_classification_cm() - refactor XLS spooling of classification_report and confusion matrix
MCCT (initial)/ Minnie Cherry Chua Tan 21-Sep-21 Updated plot_pairplot() to handle without y by using y_pred
    - Added is_target_int() thru refactoring, updated summary_stats() to handle clustering with y
MCCT (initial)/ Minnie Cherry Chua Tan 2-Oct-21 Updated set_label() to set the y_label by application
MCCT (initial)/ Minnie Cherry Chua Tan 03-Oct-21 Added the hypertuning flag - tuned
MCCT (initial)/ Minnie Cherry Chua Tan 09-Oct-21 Updated summary_stats() for cluster without n_cluster variable
MCCT (initial)/ Minnie Cherry Chua Tan 05-Nov-21 Updated the scaling of the data with test_sets
MCCT (initial)/ Minnie Cherry Chua Tan 11-Nov-21 Updated to handle the neural network of ANN, dict_to_str_keyvalue()
MCCT (initial)/ Minnie Cherry Chua Tan 14-Nov-21 Updated summary_stats() to handle the neural network statistics
MCCT (initial)/ Minnie Cherry Chua Tan 30-Jan-22 Updated N, D for keras in the init()
MCCT (initial)/ Minnie Cherry Chua Tan 11-Mar-22 Updated __init__() to check hasattr for self.X, self.y, self.data_type
MCCT (initial)/ Minnie Cherry Chua Tan 12-Mar-22 Update summary_stats() and self.file for CNN image of local directory
MCCT (initial)/ Minnie Cherry Chua Tan 06-Apr-22 Update summary_stats() for unsupervised learning to use classification_scores,
    refactor plot_cm() to be reused for non-supervised learning like SOM
MCCT (initial)/ Minnie Cherry Chua Tan 21-Apr-22 Updated plot_cm() to check for if cm_df exists to prevent errors
MCCT (initial)/ Minnie Cherry Chua Tan 29-Apr-22 Updated dict_to_str_keyvalue() to allow function within the string
MCCT (initial)/ Minnie Cherry Chua Tan 30-Apr-22 Updated dict2df() to handle None not as a string but as None
MCCT (initial)/ Minnie Cherry Chua Tan 05-May-22 Update summary_stats() to handle attribute modifier_type
MCCT (initial)/ Minnie Cherry Chua Tan 24-May-22 Update plot_cm() to put label text in a certain example
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ClassPredictionError, ROCAUC
from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction import DictVectorizer
from preprocessor import Preprocessor
from constant import Constant as c
from helper import Helper as h
from sklearn.metrics import *

import pandas as pd
import numpy as np
import re
import sys
import inspect
import time
import logging
import os

import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Model(metaclass=ABCMeta):
    @h.measure_performance
    def __init__(self, setup: dict, file: str, modifier: str,
                 prep_dict: dict, model_dict: dict, graph_dict: dict):
        """
        Function to initialize the model interface and preprocess specified data

        Parameters
        ----------
           setup (dict): configuration setup
           file (str): input file to be processed
           modifier (str): model modifier
           prep_dict (dict): preprocessor arguments
           model_dict (dict): model arguments
           graph_dict (dict): graph arguments
        """
        self.dir_out = setup[c.OUTPUT]
        self.save = setup[c.SAVE]
        self.file_stats = setup[c.FILE_STATS]
        self.graph = setup[c.GRAPH]
        self.regression_scores = setup[c.REGRESSION_SCORES]
        self.classification_scores = setup[c.CLASSIFICATION_SCORES]
        self.cluster_scores = setup[c.CLUSTER_SCORES]
        self.neural_scores = setup[c.NEURAL_SCORES]
        self.unsupervised_scores = setup[c.NEURAL_SCORES]

        self.index = setup[c.INDEX]
        self.is_last = setup[c.IS_LAST]
        self.estimator_type = setup[c.CLASSIFICATION_TYPE]

        self.modifier = modifier
        self.modifier_type = None
        self.file = file
        self.stat_file = setup[c.FILE_STATS]
        global stats_df
        if self.index == 0:
            stats_df = setup[c.STATS]

        h.debug(f'prep_dict: {prep_dict}')
        [val_df, func_df] = self.dict2df(prep_dict)
        h.debug(f'val_df: {val_df}')
        h.debug(f'func_df: {func_df}')

        self.model_func_df = None
        if len(model_dict) > 0:

            if self.modifier in c.NEURAL:
                [model_val_df, model_func_df] = self.dict2df(model_dict)
                h.debug(f'model_val_df: {model_val_df}')
                h.debug(f'model_func_df: {model_func_df}')
                self.model_func_df = model_func_df

            h.debug(f'===model: {self.dict_to_str_keyvalue(model_dict)}')
            self.model_args = model_args = self.dict_to_str_keyvalue(model_dict)
            h.debug(f'model_args: {self.model_args}')

        else:
            model_args = None
        self.graph_dict = graph_dict
        self.model_dict = model_dict
        self.prep = Preprocessor(setup, file, modifier, val_df, func_df, model_args, prep_dict, model_dict)
        self.prep.execute()
        h.debug(self.file)
        h.debug(self.prep.best_estimator)

        # get the filename for CNN in local directory based on input*.txt
        if self.modifier.lower() == c.CNN and self.prep.data_type == c.IMAGE:
            self.file = self.prep.file

        self.model_add = self.prep.model_add
        self.x_shape = self.prep.x_shape
        self.timesteps = self.prep.timesteps
        self.N, self.D = self.prep.N, self.prep.D
        if self.modifier == 'PolynomialFeatures':
            self.degree = self.prep.degree
        elif self.modifier == c.RBM:
            self.hidden = model_dict[c.HIDDEN] if c.HIDDEN in model_dict.keys() else None
            self.batch_size = model_dict[c.BATCH_SIZE] if c.BATCH_SIZE in model_dict.keys() else None
            self.epoch = model_dict[c.EPOCH] if c.EPOCH in model_dict.keys() else None


        # mapping the preprocessor variables to the model
        self.path = self.prep.path
        self.filename = self.prep.filename
        self.data_type = self.prep.data_type if hasattr(self.prep, 'data_type') else None
        self.y = self.prep.y if hasattr(self.prep, 'y') else None
        self.X = self.prep.X if hasattr(self.prep, 'X') else None
        self.ylabel = self.prep.ylabel
        self.xlabel = self.prep.xlabel
        self.xlabel_reduced = self.prep.xlabel_reduced
        self.predict_test = self.prep.predict_test
        self.predict_test_unscaled = self.prep.predict_test_unscaled
        self.ext = self.prep.ext
        self.has_y = self.prep.has_y
        self.tuned = self.prep.tuned
        self.has_scaler = self.prep.has_scaler
        self.scaler =  self.prep.scaler
        self.test_size = self.prep.test_size
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_unscaled = self.X_test_unscaled = self.y_train_unscaled =self.y_test_unscaled = None

        if self.test_size:
            self.X_train = self.prep.X_train
            self.X_test = self.prep.X_test
            self.y_train = self.prep.y_train
            self.y_test = self.prep.y_test
            self.X_train_unscaled = self.prep.X_train_unscaled
            self.X_test_unscaled = self.prep.X_test_unscaled
            self.y_train_unscaled = self.prep.y_train_unscaled
            self.y_test_unscaled = self.prep.y_test_unscaled
            h.debug(f'X_train: {self.X_train} X_test:{self.X_test} '
                  f'y_train: {self.y_train} y_test: {self.y_test}')

        self.estimator = self.prep.best_estimator
        logger.info(f'estimator: {self.estimator}')

        self.sc_X = self.prep.sc_X
        self.sc_y = self.prep.sc_y
        self.xls_file = self.prep.xls_file

        self.dataset = self.prep.dataset if hasattr(self.prep, 'dataset') else None

        if self.test_size is not None and self.test_size > 0:
            self.train_dataset = self.prep.train_dataset
            self.test_dataset = self.prep.test_dataset

        self.X_reduced =self.X_unscaled = self.X_train_unscaled = self.X_test_unscaled = None
        h.debug(f'has_y:{self.has_y} self.has_scaler: {self.has_scaler}')

        if self.has_scaler:

            self.X_reduced = self.prep.X_reduced if self.prep.X_reduced is not None else None
            if self.test_size:
                self.X_train_unscaled = self.prep.X_train_unscaled
                self.X_test_unscaled = self.prep.X_test_unscaled

                self.y_train_unscaled = self.prep.y_train_unscaled
                self.y_test_unscaled = self.prep.y_test_unscaled

            if self.sc_X:
                self.X_unscaled = self.prep.X_unscaled

            if self.sc_y:
                self.y_unscaled = self.prep.y_unscaled

        # get graph parameter
        self.title = graph_dict[c.TITLE] if c.TITLE in graph_dict else self.modifier
        h.debug(f'title: {self.title}')

        if self.estimator_type == c.REGRESSOR:
            self.smooth = graph_dict[c.SMOOTH] if c.SMOOTH in graph_dict else 0
        elif self.estimator_type == c.CLASSIFIER:
            self.smooth = graph_dict[c.SMOOTH] if c.SMOOTH in graph_dict else [10, 10]

        if self.modifier in ['PolynomialFeatures', 'SVR', 'DecisionTreeRegressor','RandomForestRegressor']:
            self.smooth = self.smooth if self.smooth else c.SMOOTH_VALUE

    @abstractmethod
    def train_predict(self):
        pass

    @abstractmethod
    def plot(self):
        pass


    def dict_to_str_keyvalue(self, model_dict: dict) -> str:
        """
        Function to transform the dict into key=value pair separated by comma

        Parameters
        ----------
           model_dict (dict): model arguments

        Return
        ------
           model_str (str): dict key=value pair separated by comma
        """
        i = 0
        n = len(model_dict)

        model_args = ''
        for k, v in model_dict.items():

            if type(v) == dict:
                continue
            if type(v) == str:
                if re.match(r"\w+\([,\w+=\w+['\".\w+]*]*\)", v):
                    model_args += k + '=' + str(v)
                else:
                    model_args += k + '="' + str(v) + '"'
            else:
                model_args += k + '=' + str(v)
            i += 1
            if (i != n):
                model_args += ','

        model_args = re.sub(r',$','', model_args)
        return model_args

    def method_param_transform(self, x):
        """
        Function to transform the dict into key=value pair separated by comma

        Parameters
        ----------
           x (dict): preprocessor method arguments

        Return
        ------
           x (str): key=value pair separated by comma
        """
        params = re.sub('{', '(', x)
        params = re.sub('}', ')', params)
        params = re.sub(r"'(\w+)':(['\s\w]+[,]*)", r"\1=\2", params)

        return params


    def dict2df(self, data:dict) -> c.PandasDataFrame:
        """
        Function to transform the dict into dataframe

        Parameters
        ----------
           data (dict): data dictionary

        Return
        ------
           value_df (dataframe): key-value pairs dataframe
           method_df (dataframe): method call dataframe
        """
        try:
            value_dict = dict(filter(lambda val: not isinstance(val[1], dict), data.items()))
            method_dict = dict(filter(lambda val: isinstance(val[1], dict), data.items()))
            h.debug(f'value: {value_dict}')
            h.debug(f'method: {method_dict}')
            value_df = pd.DataFrame({c.KEY:list(value_dict.keys()), c.VALUE:list(value_dict.values())})
            h.debug(value_df)

            method_df = pd.DataFrame()

            if len(method_dict) > 0:
                df = pd.DataFrame.from_dict(method_dict, orient='index')
                h.debug(df)

                df[c.PARAMS] = df[c.PARAMS].astype(str).apply(self.method_param_transform)
                df[c.METHOD_CALL] = df[c.METHOD] + df[c.PARAMS]
                h.debug(df[c.METHOD_CALL])
                method_df[c.KEY] = df.index
                method_df[c.VALUE] = df[c.METHOD_CALL].tolist()
                method_df[c.VALUE] = method_df[c.VALUE].replace("'None'",'None', regex=True)
                return [value_df, method_df]
            else:
                return [value_df, None]

        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)

    @h.measure_performance
    def summary_stats(self):
        """
        Function to populate the dataframe of summary statistics
        """

        i = self.index

        stats_df.iloc[i, stats_df.columns.get_loc(c.INDEX)] = i
        if self.modifier_type:
            stats_df.iloc[i, stats_df.columns.get_loc(c.MODEL)] = self.modifier + ':' + self.modifier_type
        else:
            stats_df.iloc[i, stats_df.columns.get_loc(c.MODEL)] = self.modifier
        stats_df.iloc[i, stats_df.columns.get_loc(c.FILE)] = self.file

        if hasattr(self, 'y_pred') and len(self.y_pred) > 0:
            y_pred = list(np.ravel(self.y_pred))
            #y_pred = list(self.y_pred)

        X = y_true = None

        if self.estimator_type == c.CLUSTER:
            if self.sc_X:
                X = self.X_test_unscaled.values if self.test_size else self.X_unscaled.values
            else:
                X = self.X_test.values if self.test_size else self.X.values

            if self.has_y:
                y_true = list(self.y_test) if self.test_size else list(self.y)
        else:

            if self.has_y:
                y_true = list(np.ravel(self.y_test)) if self.test_size else list(np.ravel(self.y))

        if self.estimator_type == c.REGRESSOR:
            scores = self.regression_scores
        elif self.estimator_type == c.CLASSIFIER:
            scores = self.classification_scores
        elif self.estimator_type == c.CLUSTER:
            scores = self.cluster_scores

            if self.is_target_int and self.has_y:
                scores.extend(self.classification_scores)
            a = ["calinski_harabasz_score", "davies_bouldin_score", "silhouette_score"]

            if self.no_n_cluster:
                scores = list(set(scores)-set(a))
        elif self.estimator_type == c.NEURAL_NETWORK:
            scores = self.neural_scores
            neural_keys = list(self.train.history.keys())
        elif self.estimator_type == c.UNSUPERVISED:
            scores = self.classification_scores

        nn_score = None
        for score in scores:
            logger.info(f'score: {score}')

            if self.estimator_type == c.NEURAL_NETWORK:
                nn_score = score.split('_')
                nn_score = nn_score[0]

            if self.estimator_type == c.NEURAL_NETWORK and nn_score in neural_keys:
                stats_df.iloc[i, stats_df.columns.get_loc(score)] = np.median(self.train.history[nn_score])
            elif X is not None and score in a:
                if score == 'calinski_harabasz_score':
                    stats_df.iloc[i, stats_df.columns.get_loc(score)] = calinski_harabasz_score(X, y_pred)
                elif score == 'davies_bouldin_score':
                    stats_df.iloc[i, stats_df.columns.get_loc(score)] = davies_bouldin_score(X, y_pred)
                elif score == 'silhouette_score':
                    stats_df.iloc[i, stats_df.columns.get_loc(score)] = silhouette_score(X, y_pred)
            else:

                if y_true:
                    score_func = score + '(' + str(y_true) + ',' + str(y_pred) + ')'
                    stats_df.iloc[i, stats_df.columns.get_loc(score)] = eval(score_func)

        h.debug(stats_df)
        h.debug(f'file: {self.stat_file}, last: {self.is_last}')
        if self.is_last:
            h.debug(self.stat_file)
            with pd.ExcelWriter(self.stat_file) as writer:
                stats_df.to_excel(writer, sheet_name='stats')

    #@h.measure_performance
    @property
    def is_target_int(self):
        """
        Function to check y, y_pred is int datatype
        """
        return (h.is_integer(self.y_pred) and h.is_integer(self.y))

    @h.measure_performance
    def report_classification_cm(self):
        """
        Function to populate the dataframe of summary statistics
        """
        if self.has_y:
            self.yvalue = None
            if self.is_target_int:

                y = self.y_test if self.test_size else self.y
                cm_df = pd.DataFrame(confusion_matrix(y, self.y_pred), columns=['pred_neg', 'pred_pos'],
                                     index=['neg', 'pos'])
                if self.estimator_type == c.UNSUPERVISED:
                    self.cm_df = cm_df

                self.yvalue = list(map(lambda x: 'class ' + str(x), np.unique(self.y)))
                report = classification_report(list(y), list(self.y_pred), target_names=self.yvalue, output_dict=True)
                report_df = pd.DataFrame(report).T
                with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                    cm_df.to_excel(writer, sheet_name='confusion matrix')
                    report_df.to_excel(writer, sheet_name='classification report')

    def set_label(self, type_set: str, ax: plt.Axes, title: str = None, x_label: str = None, y_label: str = None):
        """
        Function to set the label and title in the plot

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants
           ax (plt.Axes): specify axes of the plot
           title (str): title defined by the application
           x_label (str): xlabel defined by the application
           y_label (str): ylabel defined by the application
        """
        if self.modifier == 'PolynomialFeatures':
            self.title = self.title + " (degree %d" % self.degree + ")"

        if c.TRAINING_SET == type_set:
            if title:
                ax.set_title(f'{self.title} ({type_set}) {title}')
            else:
                ax.set_title(f'{self.title} ({type_set})')
        elif c.TEST_SET == type_set:
            if title:
                ax.set_title(f'({type_set}) {title}')
            else:
                ax.set_title(f'({type_set})')
        else:
            if title:
                ax.set_title(f'{self.title}  {title}')
            else:
                ax.set_title(f'{self.title}')

        if x_label:
            xlabel = x_label
        else:
            if self.test_size:
                xlabel = self.graph_dict[c.XLABEL] if c.XLABEL in self.graph_dict.keys() else self.X_train.columns[0]
            else:
                xlabel = self.graph_dict[c.XLABEL] if c.XLABEL in self.graph_dict.keys() else self.X.columns[0]
        if y_label:
            ylabel = y_label
        else:
            ylabel = self.graph_dict[c.YLABEL] if c.YLABEL in self.graph_dict.keys() else self.ylabel
        ax.set(xlabel=xlabel, ylabel=ylabel)

    @h.measure_performance
    def plot_report(self, report_type: str):
        """
        Function to plot the classification report

        Parameters
        _________
           report_type (str): set the report type  as
                CLASSIFICATION_REPORT, CONFUSION_MATRIX, PREDICTION_ERROR, ROC_AUC

        References
        __________
        https://afioto.medium.com/classification-visualizations-with-yellowbrick-d6f6150d7a32
        https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html
        https://heartbeat.fritz.ai/analyzing-machine-learning-models-with-yellowbrick-37795733f3ee
        """

        fig, ax = plt.subplots(1, 1)
        if report_type == c.CLASSIFICATION_REPORT:
            visualizer = ClassificationReport(self.classifier, classes=self.yvalue, support=True)
        elif report_type == c.CONFUSION_MATRIX:
            visualizer = ConfusionMatrix(self.classifier, classes=self.yvalue, percent=True)
        elif report_type == c.PREDICTION_ERROR:
            visualizer = ClassPredictionError(self.classifier, classes=self.yvalue)
        elif report_type == c.ROC_AUC:
            visualizer = ROCAUC(self.classifier, classes=self.yvalue)

        visualizer.ax = ax
        if self.test_size:
            visualizer.fit(self.X_train, self.y_train)
            visualizer.score(self.X_test, self.y_test)
        else:
            visualizer.fit(self.X, self.y)
            visualizer.score(self.X, self.y)
        visualizer.show()

        #fig.tight_layout()

        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_' + report_type, c.DTTM_FMT, c.GRAPHIC_EXT)
            fig.savefig(file)


    @h.measure_performance
    def plot_cm(self):
        """
        Function to plot the confusion matrix
        """
        if not hasattr(self, 'cm_df'):
            return
        if self.filename == 'Churn_Modelling':
           ax = sns.heatmap(self.cm_df, annot=True, fmt='d', cmap=plt.cm.coolwarm)
           ax.text(0.5, 0.5, '0- Not Exited, 1- Exited',  bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
           sns.heatmap(self.cm_df, annot=True, fmt='d', cmap=plt.cm.coolwarm)
        file = self.filename.split('.')[-1]
        plt.title(f"{file.capitalize()} - Confusion Matrix")
        plt.tight_layout()

        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + file + '_cm_', c.DTTM_FMT,
                                                                     c.GRAPHIC_EXT)
            plt.savefig(file)

    @h.measure_performance
    def plot_pairplot(self, type_set:str):
        """
        Function to plot the kde and pairplot to check features relationship

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants
        """
        if self.test_size:
            h.debug(f'X_train: {self.X_train_unscaled} X_test: {self.X_test_unscaled}')
            h.debug(f'unscale y_test: {self.y_test_unscaled} y_train: {self.y_train_unscaled}')
            h.debug(f'y_test: {self.y_test} y_train: {self.y_train}')
        h.debug(f'scaler X: {self.sc_X} y:{self.sc_y}')

        report_name = self.modifier + ' KDE'
        if type_set == c.TRAINING_SET:
            dataset = self.train_dataset
            report_name += ' ' + type_set

        elif type_set == c.TEST_SET:
            dataset = self.test_dataset
            report_name += ' ' + type_set

        else:
            dataset = self.dataset
            print(f'dataset: {dataset}')

        _ , n = dataset.shape
        if n > 7:
            return

        if self.has_y:
            level = len(np.unique(self.y)) + 1
            y = self.y.name
        else:
            level = len(np.unique(self.y_pred )) + 1
            y = 'y_pred'
            dataset = dataset.join(pd.DataFrame(self.y_pred, columns=[y]))
            print(dataset)

        ax = sns.pairplot(dataset, diag_kind='kde', hue=y)
        ax.map_lower(sns.kdeplot, levels=level, color=".2")
        plt.suptitle(f'{report_name}')
        report_name = report_name.replace(' ', '_')
        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(report_name, c.DTTM_FMT, c.GRAPHIC_EXT)
            plt.savefig(file)
        plt.tight_layout()
        plt.show()
