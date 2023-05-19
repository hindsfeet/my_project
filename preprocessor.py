"""
This is the preprocessor module for version 2
MCCT (initial)/ Minnie Cherry Chua Tan 10-Jul-21 Base version
MCCT (initial)/ Minnie Cherry Chua Tan 17-Jul-21 Added the static Pipeline, mime and mom-mime handdling
MCCT (initial)/ Minnie Cherry Chua Tan 23-Jul-21 Added the checking has_y in the init
MCCT (initial)/ Minnie Cherry Chua Tan 30-Jul-21 Updated execute of the numeric
MCCT (initial)/ Minnie Cherry Chua Tan 06-Aug-21 Updated execute to handle high ordinality
    and added graph of feature type
MCCT (initial)/ Minnie Cherry Chua Tan 07-Aug-21 Added assertion error handling to capture invalid
    methods or its parameter in the input file, added select_features() and hypertuning()
MCCT (initial)/ Minnie Cherry Chua Tan 08-Aug-21 Updated assertion error for the OneHotEncoder
MCCT (initial)/ Minnie Cherry Chua Tan 12-Aug-21 Updated select_features() to have excel and graphs,
    added SelectKBest handling in the select_features(), hypertune() to be not static
MCCT (initial)/ Minnie Cherry Chua Tan 13-Aug-21 Updated select_features() for the matrix decomposition
MCCT (initial)/ Minnie Cherry Chua Tan 14-Aug-21 Updated select_features() to cater for the titles,
    added handlng for the GenericUnivariate, the cardinality value
MCCT (initial)/ Minnie Cherry Chua Tan 19-Aug-21 Added measure_performance decorator, Updated
    hypertune() to have a configurable score, setup rules for cv = [3,5,10] depending on samples
MCCT (initial)/ Minnie Cherry Chua Tan 20-Aug-21 Updated to handle fetch and load of nonMIME input
    files, updated hypertune to spool the output in file
MCCT (initial)/ Minnie Cherry Chua Tan 21-Aug-21 Updated to have the RANDOM_STATE in train_test_split,
    select_features() to return DataFrame, execution() - added feature_names() and return DataFrame instead of numpy
MCCT (initial)/ Minnie Cherry Chua Tan 22-Aug-21 Refactor feature scaling, added the variables inverse_transform and reduce handling
MCCT (initial)/ Minnie Cherry Chua Tan 28-Aug-21 Update to handle PolynomialRegression()
MCCT (initial)/ Minnie Cherry Chua Tan 31-Aug-21 Update execute() to handle empty imputer, and a classifier with single parmeter
MCCT (initial)/ Minnie Cherry Chua Tan 02-Sep-21 Update to handle configurable predict_test in the input file
MCCT (initial)/ Minnie Cherry Chua Tan 06-Sep-21 Update has_scaler to reset for binary values
MCCT (initial)/ Minnie Cherry Chua Tan 14-Sep-21 Updated dataset, added train_dataset, test_dataset
MCCT (initial)/ Minnie Cherry Chua Tan 15-Sep-21 Added X to be configurable by user
MCCT (initial)/ Minnie Cherry Chua Tan 16-Sep-21 Added sns.load_dataset loading of file
MCCT (initial)/ Minnie Cherry Chua Tan 21-Sep-21 Updated to change scaling be applicable for X and y even with split
MCCT (initial)/ Minnie Cherry Chua Tan 03-Oct-21 Added the hypertuning flag - tuned
MCCT (initial)/ Minnie Cherry Chua Tan 09-Oct-21 Updated hypertune for DBSCAN since no predict method
MCCT (initial)/ Minnie Cherry Chua Tan 22-Oct-21 Added rules handling
MCCT (initial)/ Minnie Cherry Chua Tan 29-Oct-21 Added SAMPLING handling for UCB, Thompson Sampling
MCCT (initial)/ Minnie Cherry Chua Tan 05-Nov-21 Added process_tsv(), get_corpus()
MCCT (initial)/ Minnie Cherry Chua Tan 12-Nov-21 Added estimator for ANN
MCCT (initial)/ Minnie Cherry Chua Tan 29-Jan-22 Added the loading for Keras datasets load_keras_dataset()
MCCT (initial)/ Minnie Cherry Chua Tan 30-Jan-22 Debugging load_keras_dataset()
MCCT (initial)/ Minnie Cherry Chua Tan 03-Feb-22 Added load_auto_generated()
MCCT (initial) / Minnie Cherry Chua Tan 01-Mar-22 Added ENCODING handllig in read_csv in init()
MCCT (initial) / Minnie Cherry Chua Tan 02-Mar-22 Updated to passthru LabelEncder()
MCCT (initial) / Minnie Cherry Chua Tan 11-Mar-22 Added load_image_from_dir() for CNN in local directory
MCCT (initial) / Minnie Cherry Chua Tan 12-Mar-22 Updated to set the self.file for handling of local directory in CNN
MCCT (initial) / Minnie Cherry Chua Tan 17-Mar-22 Updated TRAIN_FILE to TRAIN_DIR, TEST_FILE to TEST_DIR
MCCT (initial) / Minnie Cherry Chua Tan 18-Mar-22 Added load_files(() and conditions for a dict isinstance
MCCT (initial) / Minnie Cherry Chua Tan 19-Mar-22 Added preprocess_target()
MCCT (initial) / Minnie Cherry Chua Tan 24-Mar-22 Updated preprocess_target(), added to_dataframe()
MCCT (initial) / Minnie Cherry Chua Tan 16-Apr-22 Updated load_files() for RBM handling, added process_files_by_name()
MCCT (initial) / Minnie Cherry Chua Tan 29-Apr-22 Added load_auto_generated_by()
MCCT (initial) / Minnie Cherry Chua Tan 29-Apr-22 Updated load_auto_generated_by()
MCCT (initial) / Minnie Cherry Chua Tan 04-May-22 Updated __init__() and load_auto_generated_by() to have raised option
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student

__author__ = 'Minnie Tan'

# for feature selection
import sklearn
from sklearn.feature_selection import *
from sklearn.decomposition import *

# for regression / classification
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.kernel_ridge import *
from sklearn.neural_network import *

# for cluster
from sklearn.cluster import *

# for hypertuning
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import *

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import *
from sklearn.metrics import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import category_encoders as ce
import re
import warnings
#https://botbark.com/2019/12/18/how-to-disable-warnings-in-python-and-pandas/
warnings.filterwarnings("ignore")

from constant import Constant as c
from helper import Helper as h
from sklearn.datasets import *
from sklearn.model_selection import cross_validate
from sklearn.metrics import *
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import sys
import logging
import time
import numpy as np
import os
# import torch
# import math
#import itertools

from pandas.api.types import is_numeric_dtype
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import  PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf

# MCCT is the initial name of Minnie Cherry Chua Tan - same person
__author__ = 'Minnie Tan'

logger = logging.getLogger(__name__)


class Preprocessor:
    @h.measure_performance
    def __init__(self, setup: dict, file: str, modifier: str, vdata: c.PandasDataFrame, fdata: c.PandasDataFrame, classifier_str: str, prep_dict:dict, model_dict: dict):
        """
        Function to initialize the preprocessor`

        Parameters
        ----------
           setup (dict): configuration for input directory
           file (str): input file to be processed
           modifier (str): modifier name
           vdata (DataFrame): preprocessor key-value data arguments
           fdata (DataFrame): preprocessor function data arguments
           classifier_str (str): classifier arguments
           model_dict (dict): classifier arguments in dictionary format
        """
        print("=========================")

        self.train_file = self.test_file = self.predict_file = self.test_image = None
        print(__name__)
        print(__class__.__name__)
        self.N = self.D = 1
        self.predict_test = self.predict_test_unscaled = None
        self.timesteps = self.interval = None
        self.load_keras = self.scaler = self.has_scaler = self.has_feature = self.reduced = False
        self.X = self.X_train = self.X_test = self.X_new = self.X_train_new = self.X_test_new = None
        self.xlabel = self.xlabel_reduced = self.X_reduced = self.X_unscaled = self.X_train_unscaled = self.X_test_unscaled = None
        self.y = self.y_train = self.y_test = self.y_unscaled = self.y_train_unscaled = self.y_test_unscaled = None
        self.sc_X = self.sc_y = None
        self.dataset = self.train_dataset = self.test_dataset = self.target = None
        self.best_estimator = None

        print(f'{classifier_str}')

        print(f'=========={sys._getframe(0).f_code.co_name}==========')
        print(vdata)
        test_size = h.get_value(vdata, c.TEST_SIZE)
        print(f'test_size: {test_size}')
        self.data_with_noise = False
        self.auto_type = self.data_type = None
        self.folder_in = setup[c.INPUT]
        self.estimator_type = setup[c.CLASSIFICATION_TYPE]
        self.regression_scores = setup[c.REGRESSION_SCORES]
        self.classification_scores = setup[c.CLASSIFICATION_SCORES]
        self.cluster_scores = setup[c.CLUSTER_SCORES]
        self.graph = setup[c.GRAPH]
        self.save = setup[c.SAVE]
        self.max_cardinality = setup[c.MAX_CARDINALITY]
        if test_size:
            self.test_size = test_size[0]
        elif setup[c.TEST_SIZE] >= 0:
            self.test_size = setup[c.TEST_SIZE]
        if self.test_size <= 0 or self.test_size >= 1:
            self.test_size = None
        print(f'test_size: {self.test_size}')

        self.modifier = modifier
        self.modifier_args = classifier_str
        self.model_dict = model_dict
        self.prep_dict = prep_dict
        self.val_df = vdata
        self.func_df = fdata


        has_y = h.get_value(vdata, c.HAS_Y)
        print(f'has_y: {has_y}')
        self.has_y = True if has_y is None or has_y[0] == True else False
        #self.has_y = has_y
        print(f'has_y: {self.has_y}')

        predict_test = h.get_value(vdata, c.PREDICT_TEST)
        self.predict_test = predict_test[0] if predict_test else None
        print(f'predict_test: {self.predict_test}')

        timesteps = h.get_value(vdata, c.TIMESTEPS)
        self.timesteps = 0 if timesteps is None else timesteps[0]
        print(f'timesteps: {self.timesteps}')

        interval = h.get_value(vdata, c.INTERVAL)
        self.interval = 0 if interval is None else interval[0]
        print(f'interval: {self.interval}')

        random_state = h.get_value(vdata, c.RANDOM_STATE)
        self.random_state = 0 if random_state is None else random_state[0]
        print(f'random_state: {self.random_state}')


        self.ylabel = None

        X = h.get_value(vdata, c.X)
        if X is not None:
            X = np.ravel(X)
        print(f'X: {X}')
        print(X is None)

        y = prep_dict[c.y] if c.y in prep_dict.keys() else None
        print(f'y: {y}')

        self.delimiter = delimiter = prep_dict[c.DELIMITER] if c.DELIMITER in prep_dict.keys() else None
        if delimiter:
            print('has delimiter')
        else:
            print('no delimeiter')
        self.header = header = prep_dict[c.HEADER] if c.HEADER in prep_dict.keys() else None
        print(f'header: {header}')
        self.xlabel = xlabel = prep_dict[c.XLABEL] if c.XLABEL in prep_dict.keys() else None
        print(f'label: {xlabel}')
        self.encoding = encoding = prep_dict[c.ENCODING] if c.ENCODING in prep_dict.keys() else None
        print(f'encoding: {encoding}')
        self.x_shape = x_shape = prep_dict[c.X_SHAPE] if c.X_SHAPE in prep_dict.keys() else 2
        print(f'x_shape: {x_shape}')
        self.model_add = model_add = prep_dict[c.MODEL_ADD] if c.MODEL_ADD in prep_dict.keys() else False
        print(f'model_add: {model_add}')

        columns = prep_dict[c.COLUMNS] if c.COLUMNS in prep_dict.keys() else None
        print(f'columns: {columns}')
        #print(isinstance(eval(file), dict))

        print(file)
        f = eval(file) if file.find("{") != -1 else None

        if isinstance(f, dict):
            print('file is instance of dict')
            if file.find("\\") != -1:
                self.load_image_from_dir(vdata, fdata, file)
                ext = self.data_type = c.IMAGE
            else:
                self.load_files(eval(file), X)
                if self.modifier == c.RNN:
                    self.data_type = c.LSTM
                elif self.modifier == c.RBM:
                    self.data_type = c.RBM
                ext = c.SPLIT_FILE
        else:
            print('file is not an instance of dict')
            if re.match(r"tf.keras.datasets.\w+", file):
                self.filename = file
                ext = c.KERAS
            elif file.startswith('auto_generated') or file.startswith('auto_noise'):
                print("***auto_generated")
                self.filename = file
                boundary = file.strip().split('_')[2]
                if file.startswith('auto_noise'):
                    self.data_with_noise = True

                if 'to' in boundary:
                    self.auto_type = 'to'
                    boundary = boundary.split('to')
                    lower = int(boundary[0])
                    upper = int(boundary[1])
                    h.debug(f'lower: {lower}, upper: {upper}')
                elif 'by' in boundary:
                    self.auto_type = 'by'
                    boundary = boundary.split('by')
                    number = float(boundary[0])
                    boundary = boundary[1].split('raised')
                    by = float(boundary[0])
                    if len(boundary) > 1:
                        self.auto_type = 'raised'
                        raised = float(boundary[1])
                        h.debug(f'raised: {raised}')
                    h.debug(f'number: {number}, by: {by}')
                ext = c.GENERATED
            else:
                [self.filename, ext] = h.check_ext(os.path.basename(file))

        self.path = setup[c.OUTPUT] + c.DIR_DELIM + self.filename
        # self.xls_file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.filename, c.DTTM_FMT, c.XLSX)
        self.xls_file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.XLSX)

        h.debug(f'extension: {ext}')

        self.tuned = False
        print(ext)

        self.ext = ext
        if ext == c.CSV:
            h.debug(f'with extension: {file}')
            if self.estimator_type == c.RULES:
                self.dataset = pd.read_csv(setup[c.INPUT] + '\\' + file, header = None)
                self.has_y = False
            else:
                if encoding is not None:
                    self.dataset = pd.read_csv(setup[c.INPUT] + '\\' + file, encoding=encoding)
                    self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('^Unnamed')]
                    if columns:
                        self.dataset.columns = columns
                    print(self.dataset)
                    # print(self.dataset.dropna(how='all', axis='columns'))
                    # print(self.dataset)
                else:
                    self.dataset = pd.read_csv(setup[c.INPUT] + '\\' + file)

            if self.has_y:
                h.debug('has y is empty or True')
                self.X = self.dataset.iloc[:,:-1] if X is None else self.dataset.loc[:,X]
                self.y = self.dataset.iloc[:, -1] if y is None else self.dataset.loc[:, y]

                print(f'X: {self.X}')
                print(f'y: {self.y}')
            else:
                h.debug('has y is False')
                print(X is None)
                self.X = self.dataset if X is None else self.dataset.loc[:,X]
                self.y = None
            print(f'X: {self.X}')

            if X is not None:
                self.dataset = self.X.copy()
                if self.has_y:
                    self.dataset.loc[:,self.y.name] = self.y
            print(f'dataset: {self.dataset}')
        # for text or natural language processing
        elif ext == c.TSV:
            h.debug((f'TSV: {self.prep_dict}'))
            self.data_type = c.TEXT
            self.process_tsv(setup[c.INPUT] + '\\' + file)
        elif ext == c.SPLIT_FILE:
            self.best_estimator = 'model.compile(' + self.modifier_args + ')'
            print(self.best_estimator)
        elif ext == c.IMAGE:
            self.best_estimator = 'model.compile(' + self.modifier_args + ')'
            print(self.best_estimator)
            return
        elif ext == c.KERAS:
            self.load_keras_dataset()
            return
        elif ext == c.GENERATED:

            if self.auto_type == 'to':
                self.load_auto_generated(lower, upper)
            elif self.auto_type == 'by':
                print(f'---in---: {self.data_with_noise}')
                self.load_auto_generated_by(number, by, self.data_with_noise, 1)
            elif self.auto_type == 'raised':
                print(f'noise: {self.data_with_noise} raised: {raised}')
                self.load_auto_generated_by(number, by, self.data_with_noise, raised)
            return
        elif not ext:
            h.debug(f'without extension: {file}')
            if self.has_y:
                if re.match(r'^load_dataset_\w+$', file.strip()):
                    print("@@@@@@")
                    file = re.sub(r'^load_dataset_(\w+)$',r"\1",file.strip())
                    print(f'file: {file} type(file)')
                    self.dataset = sns.load_dataset(file)
                    print(f'df: {self.dataset}')
                    if X is not None:
                        self.X = self.dataset.iloc[:, X] if isinstance(X, np.ndarray) and X.dtype in ("int32", "int64") \
                            else self.dataset.loc[:, X]
                    y = h.get_value(vdata, c.y)[0]
                    if y is not None:
                        self.y = self.dataset.iloc[:, y] if isinstance(y,int) else self.dataset.loc[:, y]
                    print(f'y:{self.y}')

                elif re.match(r'^(fetch|load)\w+$', file.strip()):
                    print(f'file: @{file}@')
                    self.X, self.y = eval(file + '(as_frame=True, return_X_y=True)')
                    if X is not None:
                        self.X = self.X.iloc[:, X] if isinstance(X,np.ndarray) and X.dtype in ("int32","int64") else self.X.loc[:, X]

                else:
                    self.X, self.y = fetch_openml(file, version=1,as_frame=True, return_X_y=True)

                    if X is not None:
                        self.X = self.X.iloc[:, X] if isinstance(X,np.ndarray) and X.dtype in ("int32","int64") else self.X.loc[:, X]
                self.dataset = self.X.join(self.y)
                print(self.dataset)

            else:
                if re.match(r'^load_dataset_\w+$', file.strip()):
                    print("@@@@@@")
                    file = re.sub(r'^load_dataset_(\w+)$',r"\1",file.strip())
                    print(f'file: {file} type(file)')
                    self.dataset = sns.load_dataset(file)
                    print(f'df: {self.dataset}')
                    if X is not None:
                        self.X = self.dataset.iloc[:, X] if isinstance(X, np.ndarray) and X.dtype in ("int32", "int64") \
                            else self.dataset.loc[:, X]
                    else:
                        self.X = self.dataset
                elif re.match(r'^(fetch|load)\w+$', file.strip()):
                    print(f'file: @{file}@')
                    self.X = eval(file + '(as_frame=True, return_X_y=False)')
                    if X is not None:
                        self.X = self.X.iloc[:, X] if isinstance(X,np.ndarray) and X.dtype in ("int32","int64") else self.X.loc[:, X]

                else:
                    self.X = fetch_openml(file, version=1,as_frame=True, return_X_y=False)
                    if X is not None:
                        self.X = self.X.iloc[:, X] if isinstance(X,np.ndarray) and X.dtype in ("int32","int64") else self.X.loc[:, X]
                self.dataset = self.X

        print(f'x: {self.X} dataset:{self.dataset}')

        self.X_df = self.X
        if not self.xlabel:
            self.xlabel = list(self.X.columns)
        print(self.xlabel)

        if self.has_y:
            if not self.ylabel:
                self.ylabel = c.y if self.y is None else self.y.name
            self.y_unscaled = self.y

    @h.measure_performance
    def rate_data(self, data: [[]], rate: int, n: int = 2) -> [[]]:
        """
        To convert the dataset

        Parameters
        ----------
            data (np.array): the dataset to be converted

        References
        ----------
        rates - https://www.udemy.com/course/deeplearning/learn/lecture/6895662#overview
        """
        print(f'rate: {rate}')
        threshold = math.floor(rate / n)
        print(f'threshold: {threshold}')
        data = torch.FloatTensor(data)
        data[data == 0] = -1
        data[data <= threshold] = 0
        data[data > threshold] = 1

        return data

    @h.measure_performance
    def process_files_by_name(self):
        """
        To preprocess file by name
        """
        if self.filename == 'movies':
            self.has_y = False
            self.xlabel = self.xlabel if self.xlabel else list(map(lambda x:'Col'+str(x),range(len(self.train_dataset.columns))))

            self.train_dataset.columns = self.xlabel
            self.test_dataset.columns = self.xlabel
            self.X = pd.concat([self.train_dataset, self.test_dataset], axis=0).reset_index(drop=True)
            self.dataset = self.X
            print(self.train_dataset)
            print(self.test_dataset)
            print(self.X)

            self.X_train = self.train_dataset
            self.X_test = self.test_dataset

    @h.measure_performance
    def load_files(self, file:dict, X:list):
        """
        To load the train / test files, assumes initially no_y

        Parameters
        ----------
            file (dict): the file directory of the train and test set
            X (list): X columns to be selected if any
        """
        self.file = self.filename = file[c.FILENAME]
        encoding = self.encoding
        delimiter = self.delimiter
        header = self.header
        print(f'file: {self.file}')
        if c.TRAIN_FILE in file:
            self.train_dataset = pd.read_csv(self.folder_in + c.DIR_DELIM + file[c.TRAIN_FILE], encoding=encoding, delimiter=delimiter)
            self.train_dataset = self.train_dataset.loc[:, ~self.train_dataset.columns.str.contains('^Unnamed')]
        if c.TEST_FILE in file:
            self.test_dataset = pd.read_csv(self.folder_in + c.DIR_DELIM + file[c.TEST_FILE], encoding=encoding, delimiter=delimiter)
            self.test_dataset = self.test_dataset.loc[:, ~self.test_dataset.columns.str.contains('^Unnamed')]

        self.train_dataset = self.train_dataset if X is None else self.train_dataset.loc[:, X]
        self.test_dataset = self.test_dataset if X is None else self.test_dataset.loc[:, X]

        print('=====')
        print(self.test_dataset)

        i = len(self.train_dataset)
        j = len(self.test_dataset)

        self.test_size = j / (i+j)
        print(f'i: {i}, j:{j}, {self.test_size}')
        if self.modifier == c.RBM:
            print('============here1================================================')
            self.process_files_by_name()
            #exit(0)
        else:
            self.X = pd.concat([self.train_dataset, self.test_dataset], axis=0).reset_index(drop=True)
        self.X_train = self.train_dataset
        self.X_test = self.test_dataset

    @h.measure_performance
    def to_dataframe(self, X: np.array, y: np.array) -> pd.DataFrame:
        """
        To change a numpy or list to a dataframe with X and y

        Parameters
        ----------
        X (np.array): X columns to be reformatted
        y (np.array): y column to be reformatted

        Returns
        _______
        (dataframe): dataset of the dataframe composed of X and y

        Reference
        ---------
        3D to 2D - https://stackoverflow.com/questions/33211988/python-reshape-3d-array-into-2d
        """
        if not isinstance(X,(list, np.ndarray)):
            return

        j = X.shape[1]

        if j > len(self.xlabel):
            xlabel = [",".join(self.xlabel) + str(i + 1) for i in range(j)]
        else:
            xlabel = self.xlabel
        ylabel = self.ylabel if self.ylabel else c.y

        if X.ndim > 2:
            X = X.transpose(2,0,1).reshape(-1,j)

        X_df = pd.DataFrame(X, columns=xlabel)
        print(f'X_df: {X_df}')
        y_df = pd.DataFrame(y, columns=[ylabel])
        print(f'y_df: {y_df}')
        dataset = pd.concat([X_df, y_df], axis=1)

        return dataset

    @h.measure_performance
    def preprocess_target(self):
        """
        To preprocess the X dataset by the timesteps, interval with specific X column to generate
        y-target and reformat X datasets

        References
        ----------
        pandas 3d - https://stackoverflow.com/questions/24290495/constructing-3d-pandas-dataframe
        """
        print(f'**y: {self.y} y_train: {self.y_train}, y_test:{self.y_test}')
        if not self.has_y:
            return

        if self.test_size:
            assert len(self.xlabel) == 1, "preprocess_target() ValueError: one value in X"
            index = 0
            if self.y_train is None:
                #print(f'X_train: {self.X_train_unscaled}, X_train: {self.X_train}')
                print(f'y_train: {self.y_train} ')
                n_Xtrain = len(self.X_train)
                print(f'index: {index}, xlabel: {self.xlabel}')
                print(f'n_Xtrain: {n_Xtrain}, timesteps: {self.timesteps}, interval: {self.interval}')

                self.X_train = self.X_train.values
                Xtrain_unscaled = self.X_train_unscaled.values
                X_train = []
                y_train = []
                X_train_unscaled = []
                y_train_unscaled = []

                for i in range(self.timesteps, n_Xtrain):
                    X_train.append(self.X_train[i - self.timesteps:i, index])
                    y_train.append(self.X_train[i, index])
                    X_train_unscaled.append(Xtrain_unscaled[i - self.timesteps:i, index])
                    y_train_unscaled.append(Xtrain_unscaled[i, index])
                X_train, y_train, y_train_unscaled = np.array(X_train), np.array(y_train), np.array(y_train_unscaled)
                X_train_unscaled, y_train_unscaled = np.array(X_train_unscaled), np.array(y_train_unscaled)

                #original code
                self.X_train = X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                self.y_train = y_train
                self.X_train_unscaled = np.reshape(X_train_unscaled, (X_train_unscaled.shape[0], X_train_unscaled.shape[1], 1))
                self.y_train_unscaled = y_train_unscaled
                print(f'X_train: {self.X_train}, {self.X_train_unscaled}')
                print(f'y_train: {self.y_train}, {self.y_train_unscaled}')
                print(f'shape X_train: {self.X_train.shape}, {self.X_train_unscaled.shape}')
                print(f'shape y_train: {self.y_train.shape}, {self.y_train_unscaled.shape}')

            if self.y_test is None:
                print(f'y_test: {self.y_test} ')
                #print(f'X_train: {self.X_test_unscaled}, X_train: {self.X_test}')
                n_Xtest = len(self.X_test)
                n = len(self.X)
                print(f'n_Xtest: {n_Xtest}, n: {n}')

                Xtest_value = self.X[n - n_Xtest - self.timesteps:].values
                Xtest_unscaled = Xtest_value = Xtest_value.reshape(-1,1)
                if self.sc_X:
                    Xtest_value = self.sc_X.transform(Xtest_value)
                X_test = []
                y_test = []
                X_test_unscaled = []
                y_test_unscaled = []
                for i in range(self.timesteps, self.timesteps + n_Xtest):
                    X_test.append(Xtest_value[i-self.timesteps:i,index])
                    y_test.append(Xtest_value[i,index])
                    X_test_unscaled.append(Xtest_unscaled[i - self.timesteps:i, index])
                    y_test_unscaled.append(Xtest_unscaled[i,index])

                X_test, y_test = np.array(X_test), np.array(y_test)
                X_test_unscaled, y_test_unscaled = np.array(X_test_unscaled), np.array(y_test_unscaled)
                # original code
                self.X_test = X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                self.y_test = y_test
                self.X_test_unscaled = np.reshape(X_test_unscaled, (X_test_unscaled.shape[0], X_test_unscaled.shape[1], 1))
                self.y_test_unscaled = y_test_unscaled

                print(f'X_test: {self.X_test}, {self.X_test_unscaled}')
                print(f'y_test: {self.y_test}, {self.y_test_unscaled}')
                print(f'shape X_test: {self.X_test.shape}, {self.X_test_unscaled.shape}')
                print(f'shape y_test: {self.y_test.shape}, {self.y_test_unscaled.shape}')

                self.X = np.concatenate([self.X_train, self.X_test], axis=0)
                self.y = np.concatenate([self.y_train, self.y_test], axis=0)
                print(f'X: {self.X}, y: {self.y}')
                print(f'shape X: {self.X.shape}, y: {self.y.shape}')
                self.X_unscaled = np.concatenate([self.X_train_unscaled, self.X_test_unscaled], axis=0)
                self.y_unscaled = np.concatenate([self.y_train_unscaled, self.y_test_unscaled], axis=0)
                print(f'X_unscaled: {self.X_unscaled}, y_unscaled: {self.y_unscaled}')
                print(f'shape X_unscaled: {self.X_unscaled.shape}, y_unscaled: {self.y_unscaled.shape}')

                self.dataset = self.to_dataframe(self.X_unscaled, self.y_unscaled)
                print(f'dataset: {self.dataset}')
                self.train_dataset = self.to_dataframe(self.X_train_unscaled, self.y_train_unscaled)
                print(f'dataset: {self.train_dataset}')
                self.test_dataset = self.to_dataframe(self.X_test_unscaled, self.y_test_unscaled)
                print(f'dataset: {self.test_dataset}')

    @h.measure_performance
    def load_target_from_dir(self, dir:str):
        """
        To load the target image from directory

        Parameters
        ----------
            dir (str): the file directory of the train and test set

        Returns
        -------
            result (list): the dependent value of the target
        """
        result = []

        for (root, dirs, file) in os.walk(dir):
            if file:
                print(f'file: {file}')
                files = [self.target.index(f.split('.')[0]) for f in file]
                result.extend(files)

        print(f'result: {result}')
        return result

    @h.measure_performance
    def load_image_from_dir(self, vdata: c.PandasDataFrame, fdata: c.PandasDataFrame, file: str):
        """
        To load the image from directory

        Parameters
        ----------
            vdata (DataFrame): preprocessor key-value data arguments
            fdata (DataFrame): preprocessor function data arguments
            file (str): the file directory of the train and test set

        References
        ----------
            os.walk, os.scandir - https://www.geeksforgeeks.org/python-list-files-in-a-directory/#:~:text=listdir()%20method%20gets%20the,it%20is%20the%20current%20directory.
                https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
            index - https://www.programiz.com/python-programming/methods/list/index
        """
        if self.modifier != c.CNN:
            return

        # 0 - no split in image, 1 - with split of train and test
        test_size = h.get_value(vdata, c.TEST_SIZE)[0]
        print(f'test_size: {test_size}')

        if test_size not in (0, 1):
            return

        print(f"file is directory: {file}")
        file = eval(file)
        print(type(file))
        self.file = self.filename = file[c.FILENAME]
        print(self.file)
        print(file[c.TRAIN_DIR])
        if c.TRAIN_DIR in file:
            self.train_dir = self.folder_in + c.DIR_DELIM + file[c.TRAIN_DIR]
            self.test_size = 1

            print(f'self.train_dir: {self.train_dir}')
            print(next(os.walk(self.train_dir))[1])
            self.target = next(os.walk(self.train_dir))[1]
            print(f'target: {self.target}')
            self.y_train = pd.DataFrame(self.load_target_from_dir(self.train_dir), columns=[c.y])
            print(f'y_train: {self.y_train}')
        if c.TEST_DIR in file:
            self.test_dir = self.folder_in + c.DIR_DELIM + file[c.TEST_DIR]
            print(f'self.test_file: {self.test_file}')
            self.y_test = pd.DataFrame(self.load_target_from_dir(self.test_dir), columns=[c.y])

        #self.has_y = True
        self.y = pd.concat([self.y_train, self.y_test], axis=0)
        print(f'self.y_train: {self.y_train}')
        print(f'self.y_test: {self.y_test}')
        print(f'self.has_y: {self.has_y}')
        print(f'self.y: {self.y}')

        if c.PREDICT_FILE in file:
            self.predict_file = self.folder_in + c.DIR_DELIM + file[c.PREDICT_FILE]
            print(f'self.predict_file: {self.predict_file}')
        print(f'file: {self.file}, filename: {self.filename}')

        assert os.path.isdir(self.train_dir), "load_image_from_dir() ValueError: Invalid train folder value"
        assert os.path.isdir(self.test_dir), "load_image_from_dir() ValueError: Invalid test folder value"
        assert os.path.isfile(self.predict_file), "load_image_from_dir() ValueError: Invalid predict file"

        print(fdata)
        func = ','.join(h.get_value(fdata, c.IMAGE))
        print(f'func: {func}')

        # to normalize or standardize the image
        rescale_val = 'rescale=1./255'
        print(f'rescale: {rescale_val}')
        func = re.sub(r'rescale\s*=\s*\d+', rescale_val, func)
        print(f'func: {func}')
        print(vdata)

        target_size = h.get_value(vdata, 'target_size')[0]
        batch_size = h.get_value(vdata, 'batch_size')[0]
        class_mode = h.get_value(vdata, 'class_mode')[0]

        print(f'target_size: {target_size},{type(target_size)}')
        print(f'batch_size: {batch_size}, {type(batch_size)}')
        print(f'class_mode: {class_mode}, {type(class_mode)}')
        print(self.predict_file)
        if self.predict_file:
            # dir = os.path.basename(os.path.dirname(self.predict_file))
            # print('===')
            # self.test_image = self.target.index(dir)
            # print(self.test_image)
            self.test_image = image.image_utils.load_img(self.predict_file, target_size=target_size)
            self.test_image = image.image_utils.img_to_array(self.test_image)
            self.test_image = np.expand_dims(self.test_image, axis=0)

        if test_size == 1:
            train_datagen = eval(func)
            self.train_dataset = train_datagen.flow_from_directory(self.train_dir,
                                                                   target_size=target_size, batch_size=batch_size,
                                                                   class_mode=class_mode)
            func = 'ImageDataGenerator(' + rescale_val + ')'
            test_datagen = eval(func)
            self.test_dataset = test_datagen.flow_from_directory(self.train_dir,
                                                                 target_size=target_size, batch_size=batch_size,
                                                                 class_mode=class_mode)

            print(self.train_dataset)
            print(self.test_dataset)
            print(self.has_y)

            if not self.has_y:
                self.X_train = self.train_dataset
                self.X_test = self.test_dataset
                print(f'X_train: {self.X_train}, y_train: {self.y_train}')
                print(f'X_test: {self.X_test}, y_test: {self.y_test}')

    @h.measure_performance
    def load_keras_dataset(self):
        """
         Function to load the Keras Dataset
        """
        data = eval(self.filename)
        (X_train, y_train), (X_test, y_test) = data.load_data()
        print(f'X_train: {X_train},  y_train: {y_train}')
        print(f'X_test: {X_test},  y_test: {y_test}')
        self.X_train, self.X_test = X_train / 255.0, X_test / 255.0
        self.y_train, self.y_test = y_train, y_test

        ntrain, self.N, self.D = X_train.shape
        ntest, self.N, self.D = X_test.shape
        print(f'N: {self.N}, D: {self.D}')

        self.test_size = ntest / (ntrain + ntest)
        print(self.test_size)

        self.best_estimator = 'model.compile(' + self.modifier_args + ')'
        self.load_keras = True
        print(np.vstack([X_train, X_test]).shape)
        print(np.hstack([y_train, y_test]).shape)
        print(np.append(X_train, X_test).shape)
        print(np.append(y_train, y_test).shape)

        self.xlabel = list(map(lambda x:'Col'+str(x), range(self.D)))
        self.ylabel = 'Target'
        self.X = np.vstack([X_train, X_test])
        self.y = np.hstack([y_train,y_test])
        n = self.y.shape[0]
        #self.y = self.y.reshape(n,1,1)

        print(self.y.shape)
        print(self.y.reshape(n,1,1))

        self.dataset = self.X
        print(self.dataset)
        return

    @h.measure_performance
    def load_auto_generated_by(self, number: float, by: float, noise: bool, raised: float):
        """
         Function to load the Auto-generated Dataset in the time series format: number(w), by(b), raised(r)
            default - np.sin(w*b)
            noise - adding noise of the default equation of + random(w) * b
            raised - np.sin((w*b)**r)
        Parameters
        ----------
            number (float): the number to be put in arange()
            by (float): the multiplier increment of the number, excluding the number
            noise (boolean): the dataset if with or without noise
            raised (float): raised to the power of np.sin(wb**2)
        """
        if noise:
            print('*************************with noise**************************')
            data = np.sin((by * np.arange(number))**raised) + np.random.randn(int(number)) * by
        else:
            data = np.sin((by*np.arange(number))**raised)

        print(self.timesteps)
        print(data)

        T = self.timesteps
        X = []
        Y = []
        for t in range(len(data) - T):
            x = data[t:t+T]
            X.append(x)
            y = data[t+T]
            Y.append(y)

        if self.x_shape == 2:
            X = np.array(X).reshape(-1,T)
        elif self.x_shape > 2:
            X = np.array(X).reshape(-1,T,1)
        Y = np.array(Y)

        N = len(X)

        print(f'shape self.X:{X.shape}, self.y:{Y.shape}')

        self.xlabel = list(map(lambda x: 'col' + str(x), range(T)))
        print(self.xlabel)
        print(f'shape X: {X.shape}, y: {Y.shape}')
        if self.x_shape == 2:
            self.X = pd.DataFrame(X, columns=self.xlabel)
        elif self.x_shape > 2:
            self.X = pd.DataFrame(X.reshape(-1,T), columns=self.xlabel)
        print(self.X)

        self.ylabel = c.y
        self.y = pd.DataFrame(Y, columns=[self.ylabel])
        self.dataset = pd.concat([self.X, self.y], axis=1)
        print(self.y)
        print(self.dataset)
        self.best_estimator = 'model.compile(' + self.modifier_args + ')'
        print(self.best_estimator)
        self.data_type = c.GENERATED

        self.X = X

        self.X_train = X[:-N//2]
        self.y_train = Y[:-N//2]
        self.X_test = X[-N//2:]
        self.y_test = Y[-N//2:]

        i = len(self.X_train)
        j = len(self.X_test)
        self.test_size = j / (i+j)
        print(f'self.test_size: {self.test_size}')

    @h.measure_performance
    def load_auto_generated(self, lower: int, upper: int, N:int=1000, dim:int=2):
        """
         Function to load the Auto-generated Dataset in uniform range

        Parameters
        ----------
            lower (int): the lower limit
            upper (int): the upper limit
            N (int): number of samples
            dim (int): dimension

         References
         __________
            random uniform with range - https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html#numpy.random.random_sample
        """
        upper = np.abs(lower) + upper
        X = upper * np.random.random_sample((N,dim)) + lower
        y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])
        print(X)
        print(f'X min: {X.min()}, max:{X.max()}, shape: {X.shape}')
        print(y)
        print(f'y min: {y.min()}, max:{y.max()}, shape: {y.shape}')

        self.xlabel = list(map(lambda x:'col'+str(x), range(dim)))
        print(self.xlabel)
        print(f'shape X: {X.shape}, y: {y.shape}')
        self.X = pd.DataFrame(X,columns=self.xlabel)
        print(self.X)
        self.ylabel = c.y
        self.y = pd.DataFrame(y, columns=[self.ylabel])
        self.dataset = pd.concat([self.X,self.y], axis=1)
        print(self.y)
        print(self.dataset)
        self.best_estimator = 'model.compile(' + self.modifier_args + ')'
        print(self.best_estimator)
        self.data_type = c.GENERATED

    @h.measure_performance
    def process_tsv(self, filename: str):
        """
         Function to process the TSV file

        Parameters
        ----------
            filename (str): TSV file
        """

        dataset = pd.read_csv(filename, delimiter='\t', quoting=3)
        stopwords = self.prep_dict[c.STOP_WORDS] if hasattr(self.prep_dict,c.STOP_WORDS) else 'english'
        print(f'stop_words: {stopwords}')
        corpus = self.get_corpus(dataset, stopwords)
        print(f'corpus: {corpus}')
        text_list = h.get_value(self.func_df, c.TEXT)
        print(text_list)
        if text_list is not None:
            text = ','.join(h.get_value(self.func_df, c.TEXT))
            print(f'**** text: {text}')

        cv = eval(text)
        self.X = pd.DataFrame(cv.fit_transform(corpus).toarray())
        self.y = dataset.iloc[:, -1]
        self.dataset = self.X.join(self.y)
        print(f'X: {self.X}')
        print(f'y: {self.y}')
        print(f'dataset: {self.dataset}')

    @h.measure_performance
    def get_corpus(self, dataset: pd.DataFrame, stop_word: str):
        """
         Function to process the text using the stop word

        Parameters
        ----------
            data (dataframe): dataframe to be checked

        Returns
        -------
            reviewed (list): reviewed corpus dataset

        Reference
        ---------
            tsv - text processing - https://www.udemy.com/course/machinelearning/learn/lecture/19958916#overview
        """

        corpus = []
        n = len(dataset)
        print(f'n:{n}')
        for i in range(n):
            review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words(stop_word)
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)
        return corpus



    def get_feature_types(self, df: c.PandasDataFrame) -> c.PandasDataFrame:
        """
         Function to get the feature_types with the corresponding counts

        Parameters
        ----------
            dataframe (dataframe): dataframe to be checked

        Returns
        -------
            features_types (dataframe): dataframe of feature type, count

        References
        ----------
            https://towardsdatascience.com/building-columntransformers-dynamically-1-6354bd08aa54
        """
        feature_types = df \
                .dtypes \
                .astype(str) \
                .value_counts() \
                .to_frame('count') \
                .rename_axis('datatype') \
                .reset_index()

        return feature_types

    def select_oh_features(self, df: c.PandasDataFrame) -> list:
        """
         Function to get columns of low cardinality satisfying the condition <= MAX_OH_CARDINALITY

        Parameters
        ----------
            df (dataframe): dataframe to be checked

        Returns
        -------
            columns (list): dataframe column names

        References
        ----------
            https://towardsdatascience.com/building-columntransformers-dynamically-1-6354bd08aa54
        """
        oh_features = \
            df \
                .select_dtypes(['object', 'category']) \
                .apply(lambda col: col.nunique()) \
                .loc[lambda x: x <= self.max_cardinality] \
                .index \
                .tolist()

        return oh_features

    def select_hc_features(self, df: c.PandasDataFrame) -> list:
        """
         Function to get columns of high cardinality satisfying the condition > MAX_OH_CARDINALITY

        Parameters
        ----------
            df (dataframe): dataframe to be checked

        Returns
        -------
            columns (list): dataframe column names

        References
        ----------
            https://towardsdatascience.com/building-columntransformers-dynamically-1-6354bd08aa54
        """

        hc_features = \
            df \
                .select_dtypes(['object', 'category']) \
                .apply(lambda col: col.nunique()) \
                .loc[lambda x: x > self.max_cardinality] \
                .index \
                .tolist()

        return hc_features


    @h.measure_performance
    def hypertune(self) -> object:
        """
        Function to hypertune the classifier parameter
        """
        try:
            classifier = eval(self.modifier + '()')
            parameters = self.model_dict
            print(f'parameters: {parameters}')

            j = self.X_train.shape[1] if self.test_size else self.X.shape[1]
            if j <= 3:
                cv = j if j > 1 else 2
            elif 4 <= j <= 9:
                cv = 5
            else:
                cv = 10
            print(f'j: {j}, cv: {cv}')

            print(classifier)
            print(type(parameters))

            grid_search = GridSearchCV(estimator=classifier,
                                       param_grid=parameters,
                                       scoring='accuracy',
                                       cv=cv,
                                       n_jobs=-1)

            X = self.X_train if self.test_size else self.X
            y = self.y_train if self.test_size else self.y

            grid_search.fit(X, y)

            print(grid_search.cv_results_)
            print(f'best score: {grid_search.best_score_}')
            print("Best Parameters: ", grid_search.best_params_)
            print(f'@best_estimator: {grid_search.best_estimator_}')

            self.best_estimator = grid_search.best_estimator_
            best_df = pd.DataFrame({c.VALUE: parameters, 'best_estimator': grid_search.best_estimator_})
            print(best_df)

            if os.path.exists(self.xls_file):
                with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                    best_df.to_excel(writer, sheet_name='GridSearchCV')
            else:
                with pd.ExcelWriter(self.xls_file) as writer:
                    best_df.to_excel(writer, sheet_name='GridSearchCV')

            self.tuned = True

            return grid_search


        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)

    @h.measure_performance
    def hypertune1(self, refit_score: str ='f1_score') -> object:
        """
        Function to hypertune the classifier parameter

        Parameters
        ----------
            refit_score (str): best metric score to be measured

        Returns
        -------
            grid_search (object): return of the hypertune function used

        Reference
        ----------
        cv - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        default int: (Stratified)KFold for classfier multiclass, else KFold for linear
        scoring - https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
        refit - https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
        """
        try:


            print( eval(self.modifier + '()'))
            classifier = eval(self.modifier + '()')


            print(f'{self.modifier_args}')
            parameters = self.model_dict
            print(f'@parameters: {parameters}', {type(parameters)})

            scores = {
                'precision_score': make_scorer(precision_score),
                'recall_score': make_scorer(recall_score),
                'accuracy_score': make_scorer(accuracy_score),
                'f1_score': make_scorer(f1_score),
                'mcc':make_scorer(matthews_corrcoef),
            }
            j = self.X_train.shape[1] if self.test_size else self.X.shape[1]
            if j <= 3:
                cv = j if j > 1 else 2
            elif 4 <= j <= 9:
                cv = 5
            else:
                cv = 10
            print(f'j: {j}, cv: {cv}')

            print(classifier)
            print(type(parameters))

            grid_search = GridSearchCV(estimator=classifier,
                                       param_grid=parameters,
                                       scoring=scores,
                                       refit=refit_score,
                                       cv=cv,
                                       n_jobs=-1)

            print(type(self.X))

            y = y_train = y_test = None

            if self.test_size:
                print(f'shape X_train: {self.X_train.shape}, X_test: {self.X_test.shape}'
                      f'y_train: {self.y_train.shape}')
                X_train = self.X_train.values if isinstance(self.X_train, (pd.DataFrame, pd.Series)) else self.X_train
                X_test = self.X_test.values if isinstance(self.X_test, (pd.DataFrame, pd.Series)) else self.X_test
                if self.has_y:
                    h.debug(f'has_y: {self.has_y}')
                    y_train = self.y_train.values if isinstance(self.y_train, (pd.DataFrame, pd.Series)) else self.y_train
                    y_test = self.y_test.values if isinstance(self.y_test, (pd.DataFrame, pd.Series)) else self.y_test

                    grid_search.fit(X_train, y_train)

                else:
                    grid_search.fit_transform(X_train)
                # make the predictions
                y_pred = grid_search.predict(X_test)

                print(f'@@ y_pred: {y_pred}, y_test: {self.y_test}, X_test: {self.X_test}')

                print(f'@@ test_size: {self.test_size}')

                print(f'\nConfusion matrix of {classifier} optimized for {refit_score} on the test data:')
                print(pd.DataFrame(confusion_matrix(self.y_test, y_pred),
                               columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
                cm_df = pd.DataFrame(confusion_matrix(self.y_test, y_pred),
                               columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
            else:
                X = self.X.values if isinstance(self.X, (pd.DataFrame, pd.Series)) else self.X

                if self.has_y:
                    y = self.y.values if isinstance(self.y, (pd.DataFrame, pd.Series)) else self.y
                    grid_search.fit(X, y)
                else:
                    grid_search.fit_transform(X)

                print('after training')
                y_pred = grid_search.predict(X)
                print(f'\nConfusion matrix of {classifier} optimized for {refit_score} on the data:')
                print(pd.DataFrame(confusion_matrix(self.y, y_pred),
                                   columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

                cm_df = pd.DataFrame(confusion_matrix(self.y, y_pred),
                                     columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
            print(cm_df)

            print(self.xls_file)
            best_df = pd.DataFrame([[refit_score,grid_search.best_score_],
                                    ['best param', grid_search.best_params_],
                                    ['best est', grid_search.best_estimator_]],
                                   columns=[c.KEY, c.VALUE])
            print(best_df)

            result = pd.concat([cm_df, best_df], ignore_index=True)
            result.to_excel(self.xls_file, sheet_name='GridSearchCV', index =False)
            print(result)
            print(f'Best params for {refit_score}: {grid_search.best_score_}')
            print(f'best param: {grid_search.best_params_}')
            print(f'@best_estimator: {grid_search.best_estimator_}')

            self.best_estimator = grid_search.best_estimator_
            print(f'{self.best_estimator}')
            print(f'@@@best_estimator: {grid_search.best_estimator_}')
            self.tuned = True
            return grid_search

        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)

    @h.measure_performance
    def select_features(self, feature: str):
        """
        Function to select the features

        Parameters
        ----------
           feature (str): method call of the feature selected in a string format

        Raises
        ------
           RuntimeError: invalid feature class name or invalid attributes

        Returns
        -------

        Reference
        ----------
        argmax components_ - higher feature importance https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
        explained_variance_ratio  as important features https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
        plot important features using components_ - last graph https://datascienceplus.com/principal-component-analysis-pca-with-python/
        - https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
        KPCA lambdas_ eigen values in decreasing order- https://iq.opengenus.org/kernal-principal-component-analysis/
        KPCA alphas_ - components_ - eigen values in the center https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA
        computation of the explained variance ratio
        https://www.mbaskool.com/business-concepts/statistics/7363-variance-ratio.html
        https://www.oreilly.com/library/view/mastering-python-for/9781789346466/d1ac368a-6890-45eb-b39c-2fa97d23d640.xhtml#:~:text=The%20explained%20variance%20score%20explains%20the%20dispersion%20of,indicating%20better%20squares%20of%20standard%20deviations%20of%20errors.
        https://stackoverflow.com/questions/29611842/scikit-learn-kernel-pca-explained-variance
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning
        """

        _, i = self.dataset.shape
        print(f'i: {i}')
        if self.has_y:
            i = i - 1
            initial_feature_names = list(self.dataset.columns[:-1])
        else:
            initial_feature_names = list(self.dataset.columns)
        print(f'initial_feature_names: {initial_feature_names}')
        print(f'2 i: {i}')
        print(f'{self.X_train}')

        if self.X_train is not None:
            _, len_X = self.X_train.shape
        else:
            _, len_X = self.X.shape
        print(f'len_x: {len_X}, i: {i}')

        assert len_X == i, "Can't do dimensionality reduction with OneHotEncoder having additional columnn"

        print(feature)
        i = feature.find('(')
        method = feature[:i]
        print(f'method: {method}')
        print(f'{method} in {dir(sklearn.decomposition)}')

        plt.clf()

        if method in dir(sklearn.decomposition):

            assert eval(feature)
            reduce_features = eval(feature)
            if self.test_size:
                self.X_train_new = reduce_features.fit_transform(self.X_train)
                self.X_test_new = reduce_features.transform(self.X_test)
                h.debug(f'reduced X_train: {self.X_train_new}, X_test: {self.X_test_new}')

            self.X_new = reduce_features.fit_transform(self.X)
            h.debug(f'X: {self.X_new}')

            h.debug(f'feature: {feature}')

            print(f'method: {method}')

            if method == 'KernelPCA':
                n = reduce_features.lambdas_.shape[0]
            elif method == 'LatentDirichletAllocation':
                n = reduce_features.exp_dirichlet_component_.shape[0]
            else:
                n = reduce_features.components_.shape[0]

            # get the scores for the top features using explaind variance ratio
            if method in ['PCA', 'IncrementalPCA', 'TruncatedSVD']:
                score = reduce_features.explained_variance_ratio_
            elif method == 'KernelPCA':
                score = reduce_features.lambdas_
            else:
                print(f'components: {reduce_features.components_}')
                #exit(0)
                #toask: partial or full fit, assumes partial fit
                if self.test_size:
                    X_hat = self.X_train_new @ reduce_features.components_
                    score = np.mean(np.sum((X_hat - self.X_train_new) ** 2, axis=1) / np.sum(self.X_train_new ** 2, axis=1))
                else:
                    X_hat = self.X_new @ reduce_features.components_
                    score = np.mean(np.sum((X_hat - self.X_new) ** 2, axis=1) / np.sum(self.X_new ** 2, axis=1))

            print(f'score: {score}')
            if method not in ['LatentDirichletAllocation', 'NMF', 'SparseCoder', 'TruncatedSVD']:

                if method == 'KernelPCA':
                    most_important = [np.abs(reduce_features.alphas_[i]).argmax() for i in range(n)]
                else:
                    print(f'{method}: {reduce_features.components_}')

                    most_important = [np.abs(reduce_features.components_[i]).argmax() for i in range(n)]
                print(f'most_important: {most_important}')

                initial_feature_names = list(self.dataset)

                #most_important_names = ['PCA{}'.format(i) +'_' + initial_feature_names[most_important[i]] for i in range(n)]
                most_important_names = [method + str(i) + '_' + initial_feature_names[most_important[i]] for i in
                                        range(n)]

                if method == 'KernelPCA':
                    top_df = pd.DataFrame(reduce_features.alphas_, columns=most_important_names)
                else:
                    print(most_important_names)
                    print(reduce_features.components_)
                    pipenv 
                    #data = reduce_features.components_
                    data = np.transpose(reduce_features.components_)
                    print(data)
                    print(data.shape)
                    print(most_important_names)
                    top_df = pd.DataFrame(data, columns=most_important_names)
                print(f'top_df: {top_df}')
                plt.clf()
                sns.heatmap(top_df, cmap='twilight').set(title=f'{method} Selected Features')
                file = h.get_path(self.path) + c.DIR_DELIM + h.get_file('Selected_features_' + method, c.DTTM_FMT, c.GRAPHIC_EXT)
                #plt.yticks(rotation=90)
                plt.tight_layout()
                plt.savefig(file)
                plt.show()
                if self.save:
                    print(f'score: {score}, most_important_names: {most_important_names}')
                    clf_metrics_df = pd.DataFrame({'feature':most_important_names, 'score':score})
                    print(f'clf_metrics_df: {clf_metrics_df}')
                    clf_metrics_df.to_excel(self.xls_file, sheet_name=method)

        elif method in dir(sklearn.feature_selection):

            print(feature)
            mode =None
            if method == 'GenericUnivariateSelect':
                feature = re.sub(r"score_func\s*=\s*'(\w+)'", r"\1", feature)
                mode =re.sub(r"\w.+mode\s*=\s'(\w+)'[,\w].+", r"\1", feature)
                print(feature)
                print(mode)

            assert eval(feature)
            reduce_features = eval(feature)

            assert self.has_y, '{method} needs to have dependent variable (y or y_train)'

            if not self.test_size:
                print(f'X: {self.X}')
                fit = reduce_features.fit(self.X, self.y)
                self.X_new = reduce_features.fit_transform(self.X, self.y)
                _, n = self.X_new.shape
                print(f'X shape: {self.X_new.shape}')
            else:
                print(f'X_train: {self.X_train}')
                print(f'@X_train: {reduce_features.fit(self.X_train, self.y_train)}')
                fit = reduce_features.fit(self.X_train, self.y_train)

                self.X_train_new = reduce_features.fit_transform(self.X_train, self.y_train)
                self.X_test_new = reduce_features.transform(self.X_test)
                print(f'X_train shape: {self.X_train_new.shape} test: {self.X_test_new.shape} ')
                _, n = self.X_train_new.shape

            print(f'scores: {fit.scores_}')
            print(f'n: {n}')
            print(type(self.X_train))
            feature_score = pd.DataFrame({'feature': self.X_df.columns, 'score':fit.scores_},
                                         index=self.X_df.columns
                                         )
            top_df = feature_score.nlargest(n,'score')

            print(f'top: {type(top_df)}')

            top_df.plot(kind='barh')
            if method == 'GenericUnivariateSelect':
                plt.suptitle(f'{method}: {mode} Selected Features')
            else:
                plt.suptitle(f'{method} Selected Features')
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file('Selected_features_' + method, c.DTTM_FMT, c.GRAPHIC_EXT)
            plt.tight_layout()
            plt.savefig(file)
            plt.show()

            if self.save:
                clf_metrics_df = top_df.reset_index(drop=True)
                print(f'clf_metrics_df: {clf_metrics_df}')

                if not mode:
                    clf_metrics_df.to_excel(self.xls_file, sheet_name=method)
                else:
                    clf_metrics_df.to_excel(self.xls_file, sheet_name=method+' '+mode)

        self.reduced = True


        print(clf_metrics_df.columns)
        X_list = list(clf_metrics_df.iloc[:, 0])
        print(f'X_list: {X_list}')

        if self.test_size:
            print(f'shape list: {len(X_list)}, X_train: {self.X_train_new.shape} X_test:{self.X_test_new.shape}')
            self.X_train = self.X_train_new = pd.DataFrame(self.X_train_new, columns=X_list)
            self.X_test = self.X_test_new = pd.DataFrame(self.X_test_new, columns=X_list)
        else:
            print(f'shape list: {len(X_list)}, x: {self.X_new.shape}')
            self.X = self.X_new = pd.DataFrame(self.X_new, columns=X_list)
        self.xlabel = X_list

        if method in dir(sklearn.decomposition):
            print('decomposition')
            X_list = np.unique([column.split('_')[1] for column in X_list])

        self.xlabel_reduced = X_list
        self.X_unscaled = self.X_reduced = self.dataset = self.dataset.loc[:, X_list]

        print(f'y: {self.y}')
        print(f'y_unscaled: {self.y_unscaled}')
        print(self.sc_y)

        self.dataset = self.dataset.join(self.y) if self.sc_y is None else self.dataset.join(self.y_unscaled)
        print(f'self.dataset: {self.dataset}')
        if self.test_size:
            print(f'X_train_unscaled: {self.X_train_unscaled} X_test_unscaled: {self.X_test_unscaled}')
            if any(self.X_train_unscaled):
                self.X_train_unscaled = self.X_train_unscaled.loc[:, X_list]

            if any(self.X_test_unscaled):
                self.X_test_unscaled = self.X_test_unscaled.loc[:, X_list]

            print("*****")
            print(f'X_train: {np.array(self.X_train)} X_test: {np.array(self.X_test)}')
            print(f'y_train: {list(self.y_train)} y_test: {list(self.y_test)}')
            if self.sc_X:
                print(f'unscale X_train: {np.array(self.X_train_unscaled)} X_test: {np.array(self.X_test_unscaled)}')
            if self.sc_y:
                print(f'unscale y_train: {list(self.y_train_unscaled)} y_test: {list(self.y_test_unscaled)}')
            print("*****")
            self.train_dataset = self.train_dataset.loc[:, X_list]
            self.train_dataset = self.train_dataset.join(self.y_train) if self.sc_y is None else self.train_dataset.join(self.y_train_unscaled)
            self.test_dataset = self.test_dataset.loc[:, X_list]
            self.test_dataset = self.test_dataset.join(self.y_test) if self.sc_y is None else self.test_dataset.join(self.y_test_unscaled)

        else:
            if any(self.X_unscaled):
                self.X_unscaled = self.X_unscaled.loc[:, X_list]
                print(f'X: {self.X_unscaled}')


    @h.measure_performance
    def feature_names(self, cat_x:list) -> list:
        """
        Function to label the category data

        Parameters
        ----------
            cat_x (list): label of the category data label in x

        Return
        ------
            features (list): the unique features in the dataset list

        Reference
        ---------
            https://www.geeksforgeeks.org/python-split-flatten-string-list/
        """
        labels = []
        for xlabel in self.X_df.columns:
            if xlabel in cat_x:
                new_list = [xlabel + '_' + item for item in np.unique(self.dataset.loc[:,xlabel])]
                labels.extend(new_list)
            else:
                labels.append(xlabel)
        return labels

    @h.measure_performance
    def execute(self):
        """
         Function to execute the preprocessing the data as an option
            - missing numeric data
            - categorical data transformation as specified
            - specify the train and test set
            - feature scaling of the data

        Parameters
        ----------

        Returns
        -------

        Reference
        ----------
        https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
        https://towardsdatascience.com/building-columntransformers-dynamically-1-6354bd08aa54
        PolynomialFeatures = https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html?highlight=polynomialfeature
        RFE - https://hersanyagci.medium.com/feature-selection-with-borutapy-rfe-and-univariate-feature-selection-f1779ca4bd01
        """
        try:

            if self.load_keras:
                return

            if self.estimator_type == c.NEURAL_NETWORK and self.data_type == c.IMAGE:
                return

            print('=======here1=======')
            if self.graph:
                feature_types = self.get_feature_types(self.X)
                print(f'feature_types: {feature_types}')
                print(type(feature_types))
                sns.barplot(x="datatype", y="count", data=feature_types).set(title="Feature Types")
                file = h.get_path(self.path) + c.DIR_DELIM + h.get_file('datatype', c.DTTM_FMT, c.GRAPHIC_EXT)
                print(f'file: {file}')
                plt.savefig(file)
                plt.show()

            print(f'estimator: {self.estimator_type}, {self.data_type}')

            if self.estimator_type == c.NEURAL_NETWORK and self.data_type == c.GENERATED:
                print('***********here***** ')
                return


            if self.data_type != c.LSTM:
                if self.has_y and not is_numeric_dtype(self.y.dtype):
                    print('y is categorical variable')
                    self.y = LabelEncoder().fit_transform(self.y)
                    print(self.y)

            if self.estimator_type == c.NEURAL_NETWORK and str(self.model_dict.values()).__contains__('Conv1D'):
                self.best_estimator = 'model.compile(' + self.modifier_args + ')'
                print('___Neural network is_text')
                return

            print(f'test_size: {self.test_size}')
            print(f'has_y: {self.has_y}')

            print('-----')
            print(self.model_dict.values())
            print(str(self.model_dict.values()).__contains__('Conv1D'))

            # 0 - variable setup
            self.X.fillna(np.nan, inplace=True)

            logger.debug(f'get_numeric_features: {h.get_numeric_features(self.X)}')
            numeric_features = h.get_numeric_features(self.X)

            logger.debug(f'get_categorical_features: {h.get_categorical_features(self.X)}')
            categorical_features = h.get_categorical_features(self.X)
            print(f'categorical_features: {categorical_features}')

            logger.debug(f'select_oh_features: {self.select_oh_features(self.X)}')
            oh_features = self.select_oh_features(self.X)
            print(f'oh_features: {oh_features}')

            logger.debug(f'select_hc_features: {self.select_hc_features(self.X)}')
            hc_features = self.select_hc_features(self.X)
            print(f'hc_features: {hc_features}')

            print(len(oh_features))

            feature = scaler = None
            if self.func_df is None:
                h.debug(f'no preprocessor arguments')
                imputer = "SimpleImputer(strategy= 'median')"
            else:
                h.debug(f'has preprocessor arguments')

                print(self.func_df)
                imputer_list = h.get_value(self.func_df, c.IMPUTER)
                print(f'imputer_list: {imputer_list}')
                imputer = ','.join(imputer_list) if imputer_list is not None else "SimpleImputer(strategy= 'median')"
                scaler_list = h.get_value(self.func_df, c.SCALER)
                print(scaler_list)

                if scaler_list is not None:
                    print('has scaler')
                    self.has_scaler = True
                    scaler = ','.join(h.get_value(self.func_df, c.SCALER))
                    print(f'**** scaler: {scaler}')

                feature_list = h.get_value(self.func_df, c.FEATURE)
                print(feature_list)
                if feature_list is not None:
                    print('has feature')
                    self.has_feature = True
                    feature = ','.join(h.get_value(self.func_df, c.FEATURE))
                    print(f'**** feature: {feature}')

            print(f'**** imputer: {imputer}')

            assert eval(imputer)

            # 1 - handling of missing values
            if self.estimator_type != c.RULES:
                if len(categorical_features) > 0 or len(oh_features) > 0 or len(hc_features) > 0:
                    numeric_pipeline = make_pipeline(eval(imputer))

                    oh_pipeline = make_pipeline(SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore'))
                    hc_pipeline = make_pipeline(ce.GLMMEncoder())
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat_oh', oh_pipeline, oh_features),
                            ('cat_hc', hc_pipeline, hc_features),
                            ('num', numeric_pipeline, numeric_features)
                            ])

                    if self.has_y:
                        self.X = preprocessor.fit_transform(self.X, self.y)
                    else:
                        self.X = preprocessor.fit_transform(self.X)
                    print(f'cat: {len(categorical_features)}, oh: {len(oh_features)}, hc:{len(hc_features)}')
                    if len(oh_features) > 0:
                        print('one hot encoder, multiple column')
                        self.xlabel = self.feature_names(oh_features)
                        self.has_feature = False
                        if self.predict_test is not None:
                            self.predict_test = pd.DataFrame(self.predict_test, columns=self.xlabel)
                    self.X = pd.DataFrame(self.X, columns=self.xlabel)

            print(f'X: {self.X}')
            print(f'y: {self.y}')

            # 2 - data split
            print('=======here2=======')
            if self.ext != c.SPLIT_FILE:
                print(f'without split in file, check the test_size ({self.test_size}) for data split')
                if self.test_size:
                    print(f'test_size: {self.test_size}, has_y: {self.has_y}, random_state: {self.random_state}')
                    print(self.X)

                    if self.has_y:
                        if self.random_state == 0:
                            self.X_train, self.X_test, self.y_train, self.y_test = \
                                train_test_split(self.X, self.y, test_size=self.test_size,
                                                 random_state=0)
                            print(type(self.X_train))

                        else:
                            self.X_train, self.X_test, self.y_train, self.y_test = \
                                train_test_split(self.X, self.y, test_size=self.test_size,
                                                 random_state=self.random_state, stratify=self.y)

                        print(f' type: {type(self.X_train)}, shape: {self.X_train.shape} = {self.X.shape}')
                        print(f'X_train: {self.X_train} X_test: {self.X_test}')
                        print(f'y_train: {self.y_train} y_test: {self.y_test}')

                        if self.data_type == c.TEXT:
                            self.y_train_unscaled = self.y_train.copy()
                            self.y_test_unscaled = self.y_test.copy()

                        self.train_dataset = self.X_train.copy()
                        self.train_dataset.loc[:, self.ylabel] = self.y_train

                        self.test_dataset = self.X_test.copy()
                        self.test_dataset.loc[:, self.ylabel] = self.y_test
                        print(self.train_dataset)
                        print(self.test_dataset)
                    else:
                        print(f'has no y')
                        i = round(len(self.X) * self.test_size)
                        print(f'i:{i} ')
                        self.X_train = self.X[:i]
                        self.X_test = self.X[i:]

                        self.train_dataset = self.X_train.copy()
                        self.test_dataset = self.X_test.copy()
                        print(f' type: {type(self.X_train)}, shape: {self.X_train.shape} = {self.X.shape}')
                        print(f'X_train: {self.X_train} X_test: {self.X_test}')

            # 3 - scale
            print('=======here3=======')

            X_list = self.xlabel if len(self.xlabel) > 0 else h.numeric_no_binary(self.X)
            print(X_list)

            self.scaler = scaler
            print(f'has scaler: {self.has_scaler} scaler: {self.scaler}')

            if self.has_scaler:

                if len(X_list) > 0:
                    self.sc_X = eval(scaler)
                    assert eval(scaler)

                if self.has_y:
                    num_list = list(filter(lambda x: x not in (0, 1), np.unique(self.y)))
                    print(f'num_list: {num_list}')
                    if len(num_list) > 0:
                        self.sc_y = eval(scaler)
                        assert eval(scaler)

                if len(X_list) == 0 and len(num_list) == 0:
                    self.has_scaler = None
                else:
                    if self.test_size:
                        self.X_train_unscaled = self.X_train.copy()
                        self.X_test_unscaled = self.X_test.copy()

                        print(self.X_train)
                        print(self.X_test)
                        print(X_list)
                        self.X_train.loc[:, X_list] = self.sc_X.fit_transform(self.X_train.loc[:, X_list])
                        self.X_test.loc[:, X_list] = self.sc_X.transform(self.X_test.loc[:, X_list])
                        print(f'====after 2  without change in the variable==')
                        print(f'X_train_unscaled: {self.X_train_unscaled}, X_test_unscaled: {self.X_test_unscaled}')
                        print(f'X_train: {self.X_train} X_test: {self.X_test}')
                        print(f'X_list: {X_list} num_list:{num_list}')
                        print(f'y_train: {self.y_train} y_test: {self.y_test}')

                        print(f'y: {self.y} y_train: {self.y_train}, y_test:{self.y_test}')
                        # exit(0)

                        if self.data_type == c.LSTM and self.ext == c.SPLIT_FILE:
                            self.preprocess_target()
                            return

                        print(f'self.has_y: {self.has_y} num_list: {len(num_list)}')

                        # has y and data is not binary
                        if self.data_type != c.LSTM and self.has_y and len(num_list) > 0:
                            print(type(self.y_train))

                            ylabel = []
                            ylabel.append(self.ylabel)
                            self.ylabel = ylabel
                            print(f'y_train: {list(self.y_train)}, y: {self.ylabel},{ylabel}')
                            print(pd.DataFrame(list(self.y_train), columns=ylabel))
                            self.y_train = self.y_train[:,np.newaxis]
                            self.y_test = self.y_test[:,np.newaxis]
                            print(pd.DataFrame(self.sc_y.fit_transform(self.y_train), columns=ylabel))
                            print(pd.DataFrame(self.sc_y.fit_transform(self.y_test), columns=ylabel))
                            self.y_train = pd.DataFrame(self.sc_y.fit_transform(self.y_train), columns=ylabel)
                            self.y_test = pd.DataFrame(self.sc_y.fit_transform(self.y_test), columns=ylabel)
                            self.y_train_unscaled = pd.DataFrame(self.sc_y.inverse_transform(self.y_train), columns=ylabel)
                            self.y_test_unscaled = pd.DataFrame(self.sc_y.inverse_transform(self.y_test), columns=ylabel)


                    #self.X_unscaled = self.X
                    if self.sc_X:
                        print('!!!!!')
                        self.X_unscaled = self.X.copy()
                        self.X.loc[:, X_list] = self.sc_X.fit_transform(self.X.loc[:, X_list])
                        print(self.X)
                        #self.X_unscaled = pd.DataFrame(self.sc_X.inverse_transform(self.X), columns=X_list)
                        print('@@@@@@!!!!!@@@@@')
                        print(f'X: {self.X} X_unscaled:  {self.X_unscaled}')

                    if self.sc_y:
                        print('@@@@@@@@@@@@')
                        print(type(self.y))
                        ylabel = []
                        ylabel.append(self.ylabel)
                        # print(f'y: {list(self.y)}, y: {self.ylabel},{ylabel}')
                        # print(pd.DataFrame(list(self.y), columns=ylabel))
                        print(self.y.shape)
                        print(self.y.values)
                        self.y_unscaled = self.y.copy()
                        self.y = self.y[:,np.newaxis]
                        self.y = pd.DataFrame(self.sc_y.fit_transform(self.y), columns=ylabel)
                        #self.y_unscaled = pd.DataFrame(self.sc_y.inverse_transform(self.y), columns=ylabel)
                        print(f'self.y: {self.y}')
                        print(f'self.y_unscaled: {self.y_unscaled}')



            print(f'x: {self.X}, {self.X_unscaled}')
            print(f'y: {self.y}, {self.y_unscaled}')

            print(f'has_feature: {self.has_feature}')
            if self.has_feature:
                self.select_features(feature)

            print(self.modifier_args)
            if self.modifier in c.NEURAL:
                self.best_estimator = 'model.compile(' + self.modifier_args + ')'
            elif self.modifier == c.RBM:
                self.best_estimator = None
            elif self.modifier == 'PolynomialFeatures':
                print(c.DEGREE in self.model_dict.keys())

                if c.DEGREE in self.model_dict.keys():
                    self.degree = self.model_dict[c.DEGREE]
                    self.best_estimator = make_pipeline(PolynomialFeatures(self.degree), LinearRegression())
                else:
                    self.degree = 2
                    self.best_estimator = make_pipeline(PolynomialFeatures(), LinearRegression())
            elif self.modifier_args is not None:
                print('====tuning====')
                compound = [True for v in self.model_dict.values() if isinstance(v, list) and len(v) > 1]
                print(len(compound))

                if len(compound) == 0:
                    print(f'unit: {self.modifier_args.replace("[","").replace("]","") }')
                    if self.estimator_type in (c.RULES, c.REINFORCEMENT_LEARNING):
                        self.best_estimator = None
                    else:
                        self.best_estimator = self.modifier + '(' + self.modifier_args.replace("[","").replace("]","") + ')'
                        self.best_estimator = eval(self.best_estimator)
                    print(f'estimator: {self.best_estimator}')

                else:
                    print('in list')

                    if self.modifier == 'DBSCAN':
                        print('if')
                        self.tuned = self.hypertune()
                    else:
                        print('else')
                        self.tuned = self.hypertune1()
                    if self.tuned:
                        print(self.tuned)
                        print(self.tuned.best_estimator_)
            else:
                print('====skip tuning====')

                self.best_estimator = self.modifier + '()'
                print(f'best_estimator: {self.best_estimator}')
                self.best_estimator = eval(self.best_estimator)

        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)

