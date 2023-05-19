"""
This is to have batch machine learning version 2.

MCCT (initial) / Minnie Cherry Chua Tan 08-Jul-21 Base version coded from scratch to update version1 to optimize code if possible
MCCT (initial) / Minnie Cherry Chua Tan 17-Jul-21 Updated to have dir to have dynamic methods of modules
MCCT (initial) / Minnie Cherry Chua Tan 06-Aug-21 Rename SPLIT_TEST to TEST_SIZE, MODEL to MODIFIER to recall easy and to avoid name conflict
MCCT (initial) / Minnie Cherry Chua Tan 07-Aug-21 Remove class Checker() and random_state variable, added assert
MCCT (initial) / Minnie Cherry Chua Tan 12-Aug-21 Combine the regression and classification
MCCT (initial) / Minnie Cherry Chua Tan 14-Aug-21 Move the config of MAX_CARDINALITY to config.json to expose to user
MCCT (initial) / Minnie Cherry Chua Tan 23-Aug-21 Updated to have get_class_object() and identify classification types
    of Regressor and Classifier, added the scores in config.json
MCCT (initial)/ Minnie Cherry Chua Tan 28-Aug-21 Update to handle PolynomialRegression()
MCCT (initial)/ Minnie Cherry Chua Tan 03-Sep-21 Update to refactor the stats_df to be global
MCCT (initial)/ Minnie Cherry Chua Tan 10-Sep-21 Update to distinguish Classifier or Regressor in sklearn.neighbors
    Update Regression to RegressionClassification as combination of Regressor and Classification distinguish by estimator_type
MCCT (initial)/ Minnie Cherry Chua Tan 14-Sep-21 Added to sklearn.naive_bayes, sklearn.tree classification with error checking
MCCT (initial)/ Minnie Cherry Chua Tan 15-Sep-21 Added clustering handling and dir(sklearn.cluster)
MCCT (initial)/ Minnie Cherry Chua Tan 20-Sep-21 Added cluster scores in check_input()
MCCT (initial) / Minnie Cherry Chua Tan 22-Oct-21 Added RULES handling of APRIORI
MCCT (initial) / Minnie Cherry Chua Tan 23-Oct-21 Added RULES handling of ECLAT
MCCT (initial) / Minnie Cherry Chua Tan 29-Oct-21 Added SAMPLING handling of UCB, Thompson Sampling
MCCT (initial) / Minnie Cherry Chua Tan 11-Nov-21 Added neural for the neural network processing
MCCT (initial) / Minnie Cherry Chua Tan 14-Nov-21 Added neural_scores
MCCT (initial) / Minnie Cherry Chua Tan 10-Mar-22 Updated check_input() to include base folder handling
MCCT (initial) / Minnie Cherry Chua Tan 02-Apr-22 Added Nonsupervised() handling for SOM and created automatically the output and log folders
"""

# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

import sklearn
import argparse
from sklearn.linear_model import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import *
from sklearn.cluster import *
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import *
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.kernel_ridge import KernelRidge
import sklearn.neural_network
#from apyori import apriori
import apyori
from sklearn_som.som import SOM

from helper import Helper as h
from constant import Constant as c
#from checker import Checker
from varname import nameof
from datetime import datetime
from regression_classification import RegressionClassification
from cluster import Cluster
from rules import Rules
from samples import Samples
from nonsupervised import Nonsupervised
from neural import Neural

import pandas as pd
import os
import json
import logging
import sys
import re

import tensorflow as tf

print(tf.__version__)


def get_log_path(path:str):
    """
     Function to create and get the log path

     Return:
        path (str): path directory of the log file
    """
    h.debug(f'path: {h.get_path(path)}')
    h.debug(f"file: {h.get_file('log', c.DT_FMT, 'txt')}")

    return h.get_path(path) + c.DIR_DELIM + h.get_file('log', c.DT_FMT, 'txt')

def check_input():
    """
    Function to construct the argument parser and parse the arguments

    Parameters
    ----------
        dir_in (str): input directory of the data files
        dir_out (str): output directory to contain the graphs and results of the data modelling
        input (DataFrame): dataset of Model:File:PreprocessorArguments:ModelArguments
        file_stats (str): name of the output summary statistical file

    Returns
    -------
        setup (dict): global setup
        data (df obj): input file data
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="Path to the input file")
    args = vars(ap.parse_args())

    config = pd.read_json(c.CONFIG_FILE)

    folder_base = config[c.FOLDER][0][c.BASE]
    folder_in = config[c.FOLDER][0][c.INPUT]
    folder_out = config[c.FOLDER][0][c.OUTPUT]
    folder_log = config[c.FOLDER][0][c.LOG]
    test_size = config[c.DETAIL][0][c.TEST_SIZE]
    save_on = config[c.DETAIL][0][c.SAVE]
    graph_on = config[c.DETAIL][0][c.GRAPH]
    h.debug(f' max_cardinality: {config[c.DETAIL][0][c.MAX_CARDINALITY]}')
    max_cardinality = config[c.DETAIL][0][c.MAX_CARDINALITY]
    regression_scores = config[c.DETAIL][0][c.REGRESSION_SCORES]
    classification_scores = config[c.DETAIL][0][c.CLASSIFICATION_SCORES]
    cluster_scores = config[c.DETAIL][0][c.CLUSTER_SCORES]
    neural_scores = config[c.DETAIL][0][c.NEURAL_SCORES]

    assert 0 <= test_size <= 1, "config.json ValueError: Invalid test_size range(0,1)"
    assert isinstance(save_on, bool), "config.json ValueError: Invalid save_on bool value"
    assert isinstance(graph_on, bool), "config.json ValueError: Invalid graph_on bool value"
    assert isinstance(max_cardinality, int), "config.json ValueError: Invalid max_cardinality int value"
    assert isinstance(regression_scores, list), "config.json TypeError: Error: Invalid data type of regression scores list"
    assert isinstance(classification_scores, list), "config.json TypeError: Error: Invalid data type of classification scores list"
    assert isinstance(cluster_scores, list), "config.json TypeError: Error: Invalid data type of cluster scores list"
    assert isinstance(neural_scores, list), "config.json TypeError: Error: Invalid data type of neural scores list"
    assert h.sublist_in_list(regression_scores, dir(sklearn.metrics)), "config.json ValueError: Invalid value in regression scores, value undefined in sklearn.metrics"
    assert h.sublist_in_list(classification_scores, dir(sklearn.metrics)), "config.json ValueError: Invalid value in classification scores, value undefined in sklearn.metrics"
    assert h.sublist_in_list(cluster_scores, dir(sklearn.metrics)), "config.json ValueError: Invalid value in cluster scores, value undefined in sklearn.metrics"

    if save_on:
        file_stats = config[c.DETAIL][0][c.FILE_STATS]
    assert isinstance(file_stats, str), "config.json ValueError: Invalid file_stats str value"

    if folder_base == '.':
        folder_base = os.getcwd()

    assert os.path.isdir(folder_base), "config.json ValueError: Invalid folder base value"

    folder_in = folder_base + c.DIR_DELIM + folder_in
    folder_out = folder_base + c.DIR_DELIM + folder_out
    folder_log = folder_base + c.DIR_DELIM + folder_log

    folder_out = h.get_path(folder_out)
    folder_log = h.get_path(folder_log)

    assert os.path.isdir(folder_in), "config.json ValueError: Invalid folder in value"
    assert os.path.isdir(folder_out), "config.json ValueError: Invalid folder out value"
    assert os.path.isdir(folder_log), "config.json ValueError: Invalid folder log value"

    h.debug(f'folder_in: {folder_in}')
    h.debug(f'folder_out: {folder_out}')
    h.debug(f'folder_log: {folder_log}')

    os.chdir(folder_in)

    data = pd.read_csv(args[c.INPUT],names=[c.MODEL, c.FILE, c.PREP_ARGS, c.MODEL_ARGS, c.GRAPH_ARGS],
        skiprows = 1, sep = c.PIPE)

    setup = {}
    setup[c.BASE] = folder_base
    setup[c.INPUT] = folder_in
    setup[c.OUTPUT] = folder_out
    setup[c.LOG] = folder_log
    setup[c.SAVE] = save_on
    setup[c.GRAPH] = graph_on
    setup[c.FILE_STATS] = h.get_path(folder_out) + c.DIR_DELIM + h.get_file(file_stats, c.DTTM_FMT, c.XLSX)
    setup[c.TEST_SIZE] = test_size
    setup[c.MAX_CARDINALITY] = max_cardinality
    setup[c.REGRESSION_SCORES] = regression_scores
    setup[c.CLASSIFICATION_SCORES] = classification_scores
    setup[c.CLUSTER_SCORES] = cluster_scores
    setup[c.NEURAL_SCORES] = neural_scores

    h.debug(f'setup: {setup}')
    return [setup, data]


def process_input(setup: dict, input: c.PandasDataFrame, logger: logging.Logger):
    """
    Function to process the data modelling including the preproessing

    Parameters
    ----------
        setup (dict): config object based on config.json
        input (DataFrame): dataset of Model|File|PreprocessorArguments|ModelArguments

    Raises
    ------
        RuntimeError: TypeError Invalid classifier or attributes

    References
    ----------
        Segregate Regressor / Classifier - https://scikit-learn.org/stable/computing/scaling_strategies.html
        regex exclude word - https://regexland.com/regex-match-all-except/
        Dataframe dtypes - # https:///pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
    """

    try:
        n = len(input)

        lm_list = h.get_class_object(dir(sklearn.linear_model))
        kr_list = h.get_class_object(dir(sklearn.kernel_ridge))
        svm_list = h.get_class_object(dir(sklearn.svm))
        e_list = h.get_class_object(dir(sklearn.ensemble))
        nb_list = h.get_class_object(dir(sklearn.naive_bayes))
        nn_list = h.get_class_object(dir(sklearn.neural_network))
        n_list = h.get_class_object(dir(sklearn.neighbors))
        t_list = h.get_class_object(dir(sklearn.tree))
        c_list = h.get_class_object(dir(sklearn.cluster))
        a_list = dir(apyori)

        columns = [c.INDEX, c.MODEL, c.FILE]
        columns.extend(setup[c.REGRESSION_SCORES])
        columns.extend(setup[c.NEURAL_SCORES])
        columns.extend(setup[c.CLASSIFICATION_SCORES])
        columns.extend(setup[c.CLUSTER_SCORES])
        columns = list(dict.fromkeys(columns))

        #global stats_df
        stats_df = pd.DataFrame(columns=columns, index=range(n))

        setup[c.STATS] = stats_df

        for i in range(n):

            modifier = input.iloc[i, 0]
            file = input.iloc[i, 1]
            assert file is not None, "ValueError: No input file"
            prep_dict = h.str_to_dict(input.iloc[i, 2])
            modifier_dict = h.str_to_dict(input.iloc[i, 3])
            graph_dict = h.str_to_dict(input.iloc[i, 4])

            logger.info(f'dict preprocessor: {prep_dict}, modifier: {modifier_dict}, graph:{graph_dict}')
            logger.info(f'modifier: {modifier}')

            setup[c.INDEX] = i
            setup[c.IS_LAST] = i + 1 == n
            setup[c.LAST] = n
            h.debug(f'i: {i + 1}, last record: {setup[c.IS_LAST]}')
            h.debug(f'modifier: {modifier}')

            if modifier in lm_list or modifier in kr_list or modifier in svm_list or modifier in e_list \
                    or modifier in n_list or modifier in nb_list or modifier in nn_list or modifier in t_list or \
                    modifier in 'PolynomialFeatures':
                h.debug('Regressor or Classifier')

                # both - KernelRidge, OneClassSVM, are under Classifier
                # https://scikit-learn.org/stable/computing/scaling_strategies.html?highlight=minmaxscaler
                if (modifier in lm_list and (re.match(r"\w*Classifier\w*", modifier) or modifier in ['Perceptron','LogisticRegression'])) or \
                    (modifier in kr_list) or \
                    (modifier in svm_list and re.match(r"\w*[CM]", modifier)) or \
                    (modifier in n_list and re.match(r"^(?!.*(Regressor)).*", modifier)) or \
                    (modifier in nb_list and (re.match(r"\w*NB", modifier))) or \
                    (modifier in t_list and (re.match(r"\w*Classifier", modifier))) or \
                    (modifier in e_list and (re.match(r"\w*Classifier", modifier) or modifier in ['IsolationForest', 'RandomTreesEmbedding'])) or \
                    (modifier in nn_list and (re.match(r"\w*Classifier", modifier) or modifier in ['BernoulliRBM'])):
                    h.debug(f"Classifier: {modifier}")
                    setup[c.CLASSIFICATION_TYPE] = c.CLASSIFIER
                elif (modifier in nb_list and re.match(r"^(?!.*(NB)).*", modifier)) or \
                    (modifier in t_list and re.match(r"^(?!.*(Classifier|Regressor)).*", modifier)):
                    print(f'invalid classifier: {modifier}')
                    continue
                else:
                    h.debug(f"Regressor: {modifier}")
                    setup[c.CLASSIFICATION_TYPE] = c.REGRESSOR

                estimator = RegressionClassification(setup, file, modifier, prep_dict, modifier_dict, graph_dict)
                estimator.execute()
            elif modifier in c_list:
                logger.info(f"Cluster: {modifier}")
                setup[c.CLASSIFICATION_TYPE] = c.CLUSTER
                estimator = Cluster(setup, file, modifier, prep_dict, modifier_dict, graph_dict)
                estimator.execute()
            elif modifier in a_list or modifier == c.ECLAT:
                logger.info(f'Rules: {modifier}')
                setup[c.CLASSIFICATION_TYPE] = c.RULES
                estimator = Rules(setup, file, modifier, prep_dict, modifier_dict, graph_dict)
                estimator.execute()
            elif modifier in c.SAMPLING:
                h.debug(f'Sampling: {modifier}')
                setup[c.CLASSIFICATION_TYPE] = c.REINFORCEMENT_LEARNING
                estimator = Samples(setup, file, modifier, prep_dict, modifier_dict, graph_dict)
                estimator.execute()
            elif modifier in c.NONSUPERVISED:
                h.debug(f'Unsupervised: {modifier}')
                setup[c.CLASSIFICATION_TYPE] = c.UNSUPERVISED
                h.debug(f'File: {file}, {type(file)}')
                estimator = Nonsupervised(setup, file, modifier, prep_dict, modifier_dict, graph_dict)
                estimator.execute()
            elif modifier in c.NEURAL:
                h.debug(f'Neural Network: {modifier}')
                setup[c.CLASSIFICATION_TYPE] = c.NEURAL_NETWORK
                h.debug(f'File: {file}, {type(file)}')
                estimator = Neural(setup, file, modifier, prep_dict, modifier_dict, graph_dict)
                estimator.execute()

    except (Exception) as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        logger.error(message)


# Run the main script
if __name__ == '__main__':
    model = 'SVC'

    [setup, input] = check_input()
    print(f'setup: {setup}')
    print(f'log: {setup[c.LOG]}')
    log_file = get_log_path(setup[c.LOG])
    print(f'log file: {log_file}')

    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        level=logging.DEBUG,
                        filename=log_file)

    setup[c.LOGGING] = logging
    logger = logging.getLogger(__name__)

    logger.info(f'!!!In: {setup[c.INPUT]}, Out: {setup[c.OUTPUT]}')
    process_input(setup, input, logger)