"""
This is for helper tool implementation

MCCT (initial) / Minnie Cherry Chua Tan 10-Jul-21 Base version coded for upgrade to version 2
MCCT (initial) / Minnie Cherry Chua Tan 16-Jul-21 Added get_path(), get_file()
MCCT (initial) / Minnie Cherry Chua Tan 17-Jul-21 Added debug()
MCCT (initial) / Minnie Cherry Chua Tan 30-Jul-21 Added get_value()
MCCT (initial) / Minnie Cherry Chua Tan 06-Aug-21 Updated the check_ext() to have the file name returned
MCCT (initial) / Minnie Cherry Chua Tan 19-Aug-21 Added @measure_performance decorator
MCCT (initial) / Minnie Cherry Chua Tan 23-Aug-21 Added get_class_object() and sublist_in_list()
MCCT (initial) / Minnie Cherry Chua Tan 30-Aug-21 Updated get_value() to handle empty dataframe of a given c.KEY
MCCT (initial) / Minnie Cherry Chua Tan 03-Oct-21 Added is_integer()
MCCT (initial) / Minnie Cherry Chua Tan 10-Feb-22 Updated is_integer() to add the subtype of int
MCCT (initial) / Minnie Cherry Chua Tan 31-Mar-22 Added to_values()
MCCT (initial) / Minnie Cherry Chua Tan 08-Apr-22 Added pd_to_values() from refactoring to_values()
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'


from constant import Constant as c
from sklearn.compose import make_column_selector
from datetime import datetime
import pandas as pd
import json
import os
import sys
import numpy as np
from functools import wraps
import tracemalloc
from time import perf_counter
import re

class Helper:
    @classmethod
    def str_to_dict(self, data: str) -> dict:
        """
         Function to convert str to dict

         Parameters
         __________
            dataset (pd.DataFrame): DataFrame to be checked

         Return
         ______
            data (dict): dictionary of the data
        """
        if pd.isna(data) or not data:
            return {}
        else:
            return json.loads(data)

    @classmethod
    def is_integer(self, data: object) -> bool:
        """
         Function to check if object is_integer

         Parameters
         __________
            data (object): data to be checked in numpy and pandas form

         Return
         ______
            is_integer (bool): true if the object is all integer, else false

         References
         __________
               np.issubdtype - https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not
               int, real (not working for usigned) - https://stackoverflow.com/questions/66283951/how-to-use-isinstance-to-test-all-possible-integer-types
        """
        return (isinstance(data, (pd.DataFrame, pd.Series)) and np.issubdtype(data.dtypes, np.integer)) or \
               (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer))

    @classmethod
    def get_missing_cols(self, dataset: pd.DataFrame) -> list:
        """
         Function to get the missing columns

         Parameters
         __________
            dataset (pd.DataFrame): DataFrame to be checked

         Return
         ______
            missing (list): the numeric missing column list
        """

        labels = dataset.select_dtypes(include=["number"]).columns
        missing = [l for l in labels if len(dataset[dataset[l].isnull()]) > 0]

        return missing

    @classmethod
    def pd_to_values(self, data: pd.DataFrame) -> list:
        """
         Function to convert the values if pd.Dataframe or pd.Series

         Parameters
         __________
            data (pd.DataFrame): values to be checked

         Return
         ______
            values (list): the numpy values
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        return data

    @classmethod
    def to_values(self, data: pd.DataFrame) -> list:
        """
         Function to convert the values into 1D numpy or list

         Parameters
         __________
            data (pd.DataFrame): values to be checked

         Return
         ______
            values (list): the 1D values
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        data = self.pd_to_values(data)
        return np.ravel(data)

    @classmethod
    def get_numeric_index(self, dataset: pd.DataFrame) -> list:
        """
         Function to get the numeric columns

         Parameters
         __________
            dataset (pd.DataFrame): DataFrame to be checked

         Return
         ______
            index (list): the numeric column list index
        """
        return (dataset.applymap(type) == ["float64", "int64"]).all(0)

    @classmethod
    def get_numeric_features(self, dataset: pd.DataFrame) -> list:
        """
         Function to get the numeric columns

         Parameters
         __________
            dataset (pd.DataFrame): DataFrame to be checked

         Return
         ______
            labels (list): the numeric column list
        """

        labels = dataset.select_dtypes(include=["number"]).columns

        return list(labels)

    @classmethod
    def get_categorical_features(self, dataset: pd.DataFrame) -> list:
        """
         Function to get the categorical columns

         Parameters
         __________
            dataset (pd.DataFrame): DataFrame to be checked

         Return
         ______
            labels (list): the str column list
        """

        labels = dataset.select_dtypes(include=["object", "category"]).columns
        return list(labels)

    @classmethod
    def get_path(self, path: str) -> str:
        """
         Function to create and get the path

         Parameters
         __________
            path (str): directory path

         Return
         ______
            path (str): path directory of the file
        """
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    @classmethod
    def get_file(self, file_prefix:str, date_format:str, ext:str) -> str:
        """
         Function to generate the file string
            <file_prefix>_<date_format>.<ext>

         Parameters
         __________
            file_prefix (str): file prefix
            date_format (str): date format
            ext (str): file extension

         Return
         ______
            file (str): file string
        """
        now = datetime.now()
        date_str = now.strftime(date_format)

        return file_prefix + '_' + date_str + '.'+ ext

    @classmethod
    def debug(self, message):
        """
         Function to debug in console
            <file> func=<function> line=<line_no> message
            https://stackoverflow.com/questions/6810999/how-to-determine-file-function-and-line-number

         Parameters
         __________
            file_prefix (str): file prefix
        """
        import inspect
        callerframerecord = inspect.stack()[1]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)

        print(info.filename, 'func=%s' % info.function, 'line=%s:' % info.lineno, message)

    @classmethod
    def check_ext(self, file:str) -> list:
        """
         Function to check if there's file extension

         Parameters
         ----------
            file (str): file name

         Return
         ------
            name (str): filename
            ext (str): extension name
        """
        name, extension = os.path.splitext(file)
        return [name, extension[1:]]

    @classmethod
    def measure_performance(self, func:object) -> object:
        """
         Measure performance of a function

         Parameters
         ----------
            func (object): function wrapper
         Return
         ------
            wrapper (object): function wrapper

        Reference
        ---------
            measure_performance() - https://www.freecodecamp.org/news/python-decorators-explained-with-examples/
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            print(f'{"-" * 40}')
            start_time = perf_counter()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            finish_time = perf_counter()
            print(f'Function: {func.__name__}')
            print(f'Memory usage:\t\t {current / 10 ** 6:.6f} MB \n'
                  f'Peak memory usage:\t {peak / 10 ** 6:.6f} MB ')
            print(f'Time elapsed is seconds: {finish_time - start_time:.6f}')
            print(f'{"-" * 40}')
            tracemalloc.stop()
            return result
        return wrapper

    @classmethod
    def get_value(self, df: object, key: str) -> object:
        """
         Function to get the value of the key having [key, value] columns of the dataframe

         Parameters
         __________
            df (object): dataframe with column [0,1] = [key,value]
            key (str): key of the dataframe

         Return
         ______
            value (object): value of the key
        """

        selected_df = df[df.iloc[:,0] == key]
        print(f'selected_df: {selected_df}')
        print(f'None: {selected_df is None}')
        print(f'len: {len(selected_df) ==0}')
        print(f'empty: {selected_df.empty}')
        if selected_df.empty:
            return None

        value = selected_df.iloc[:,1]
        if not value.empty:
            return sorted(value.values)

    @classmethod
    def numeric_no_binary(self, df: pd.DataFrame) -> list:
        """
         Function to get only numeric columns from DataFrame excluding bool or binary
         as its a subclass of int64

         Parameters
         __________
            df (pd.DataFrame): DataFrame to be checked

         Return
         ______
            cols (list): the numeric column names excluding bool or binary type
        """
        cols = df.select_dtypes(include=["number"]).columns.tolist()
        new_cols = []
        for col in cols:
            x_list = np.unique(df[col])
            num_list = list(filter(lambda x: x not in (0, 1) and not isinstance(x, bool), x_list))
            if num_list:
                new_cols += [col]
        return new_cols

    @classmethod
    def get_class_object(self, l: list) -> list:
        """
         Function to get class object in the default Python module list

         Parameters
         __________
            l (list): Python module list returned by dir()

         Return
         ______
            data (list): class object list
        """
        data = [i for i in l if re.match(r'[A-Z]+',i)]

        return data

    @classmethod
    def sublist_in_list(self, sublist: list, l: list) -> bool:
        """
         Function to check whether the sublist is in list

         Parameters
         __________
            sublist (list): sublist to be verified
            l (list): main list

         Return
         ______
            (bool): True: sublist is in list else False

        Reference
        ---------
            sublist check - https://www.geeksforgeeks.org/python-check-if-one-list-is-subset-of-other/
        """

        return set(sublist).issubset(set(l))