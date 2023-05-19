"""
This is for neural network model implementation

MCCT (initial)/ Minnie Cherry Chua Tan 11-Nov-21 Added skeleton of train_predict(), plot(), execute()
MCCT (initial)/ Minnie Cherry Chua Tan 12-Nov-21 Added contents of train_predict()
MCCT (initial)/ Minnie Cherry Chua Tan 13-Nov-21 Added save_xlsx(), and updated train_predict()
MCCT (initial)/ Minnie Cherry Chua Tan 14-Nov-21 Added save_stats(), plot(), updated save_xlsx()
MCCT (initial)/ Minnie Cherry Chua Tan 22-Jan-22 Updated the plot() to add test validation data if applicable with legend
MCCT (initial)/ Minnie Cherry Chua Tan 30-Jan-22 Updated the train_predict() to add Flatten special handling in the model
MCCT (initial)/ Minnie Cherry Chua Tan 31-Jan-22 Added plot_confusion_matrix()
MCCT (initial)/ Minnie Cherry Chua Tan 04-Feb-22 Updated plot_confusion_matrix() - to include y instead of y_test only;
    updated plot() - to check for discrete and continuous values, added plot_3d()
MCCT (initial)/ Minnie Cherry Chua Tan 08-Feb-22 Added CNN handling for train_predict(),
    updated to remove Flatten() as the handling is change to the input file instead
MCCT (initial)/ Minnie Cherry Chua Tan 09-Feb-22 Updated for CNN handling purposes
MCCT (initial)/ Minnie Cherry Chua Tan 10-Feb-22 Updated plot() to update the check if target is integer
MCCT (initial)/ Minnie Cherry Chua Tan 11-Feb-22 Updated plot_confusion_matrix() to add a graph of misclassified example
MCCT (initial)/ Minnie Cherry Chua Tan 02-Mar-22 Added is_text to check for text preprocessing
MCCT (initial)/ Minnie Cherry Chua Tan 03-Mar-22 Added text_preprocess() and a CNN natural language processing of tensorflow
MCCT (initial)/ Minnie Cherry Chua Tan 09-Mar-22 Updated train_predict() to have correct y_pred for 2 targets (y)
MCCT (initial)/ Minnie Cherry Chua Tan 11-Mar-22 Updated to add handling of local directory and without y target
MCCT (initial)/ Minnie Cherry Chua Tan 12-Mar-22 Updated to set n=2 for binomial, check for has_y, refactor codes for confusedexample to plot_confusion_example()
MCCT (initial)/ Minnie Cherry Chua Tan 17-Mar-22 Updated save_xlsx() to add sheet for image prediction in the local directory
MCCT (initial)/ Minnie Cherry Chua Tan 18-Mar-22 Updated save_xlsx() to add the y-true and y-pred
MCCT (initial)/ Minnie Cherry Chua Tan 21-Mar-22, 22-Mar-22 Updated save_xlsx() to cater for specific condition of y_true value
MCCT (initial)/ Minnie Cherry Chua Tan 24-Mar-22 Updated train_predict() and save_xlsx() for LSTM
MCCT (initial)/ Minnie Cherry Chua Tan 25-Mar-22 Added plot_target()
MCCT (initial)/ Minnie Cherry Chua Tan 06-Apr-22 Refactor confusion_matrix to plot_cm()
MCCT (initial)/ Minnie Cherry Chua Tan 15-Apr-22 Updated save_xlsx() to safeguard, just additional checking
MCCT (initial)/ Minnie Cherry Chua Tan 29-Apr-22 Updated train_predict() to simplify adding of the model
MCCT (initial)/ Minnie Cherry Chua Tan 30-Apr-22 Added multistep_forecast(), its handling, set target_continous
MCCT (initial)/ Minnie Cherry Chua Tan 04-May-22 Updated the title for the lineplot(), updated train_predict() to handle Ds of RNN
MCCT (initial)/ Minnie Cherry Chua Tan 20-May-22 21-May-22 Updated the filename to ANN if the built model has Dense only,
                                                 updated multistep_forecast() for ANN
MCCT (initial)/ Minnie Cherry Chua Tan 24-May-22 Updated plot_confusion_example() to add the index in the title
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

#import tensorflow


from model import Model
from constant import Constant as c
from helper import Helper as h
from functools import partial
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import *
#from tensorflow.keras.models import *
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import re
import os

logger = logging.getLogger(__name__)

class Neural(Model):

    @h.measure_performance
    def train_predict(self):
        """
        Function to train and predict the model using the datasets

        References
        ----------
        history - https://datascience.stackexchange.com/questions/74742/tensorflow-keras-fit-accuracy-and-loss-both-increasing-drastically
        evaluate, predict - https://www.circuitbasics.com/neural-networks-in-python-ann/
        plot,tuning - https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
        summary print - https://stackoverflow.com/questions/63843093/neural-network-summary-to-dataframe
        """
        logger.info(f'train and predict: {self.modifier.lower()}')

        h.debug(f'model_dict: {self.model_dict}')
        h.debug(f'model_func_df: {self.model_func_df}')

        self.modifier_type = None

        if self.modifier.lower() in c.NEURAL:

            # not the image from local directory
            if self.modifier.lower() == c.CNN and self.data_type != c.IMAGE:

                is_text = len(self.model_func_df[self.model_func_df[c.VALUE].str.contains('^Conv1D')]) > 0
                h.debug(is_text)

                if is_text:
                    [V, T, D] = self.text_preprocess()
                    h.debug(f'V: {V}, T: {T}, D: {D}')
                    K = 1
                else:
                    X_train = np.expand_dims(self.X_train,-1)
                    X_test = np.expand_dims(self.X_test,-1)
                    K = len(set(self.y_train))


            # Initial the neural network
            model = tf.keras.models.Sequential()
            self.model_func_df = self.model_func_df.sort_values(by=c.KEY)

            self.model_func_df[c.KEY] = self.model_func_df[c.KEY].str.lower()
            df = self.model_func_df[self.model_func_df[c.KEY].str.startswith('add')]

            # not the image from local directory
            if self.modifier.lower() == c.CNN and self.data_type != c.IMAGE:
                i, n = 0, len(df) - 1

                for v in df[c.VALUE]:
                    h.debug(f'i: {i}')
                    h.debug(f'v:{v}')
                    if i == 0:
                        if is_text:
                            start = Input(shape=(T,))
                            next = Embedding(V+1, D)(start)
                            next = eval(v)(next)
                        else:
                            start = Input(shape=X_train[0].shape)
                            next = eval(v)(start)
                    elif i < n:
                        next = eval(v)(next)
                    else:
                        add = 'add' + str(i+1)
                        activation = self.model_dict[add][c.PARAMS]['activation']
                        end = Dense(units=K, activation=activation)(next)
                        model = tf.keras.models.Model(start, end)
                    i += 1
            elif self.model_add:
                i = 0
                for v in df[c.VALUE]:

                    if i == 0:
                        T = self.X.shape[1]
                        if self.modifier == c.RNN:
                            self.modifier_type = v.split('(')[0]
                            if self.modifier_type == 'Dense':
                                self.modifier = c.ANN
                                self.modifier_type = None
                                self.xls_file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.XLSX)
                            h.debug(f'modifier_type: {self.modifier_type}')
                            if self.modifier_type in ['SimpleRNN','GRU','LSTM']:
                                self.X = self.X.reshape(-1,T,1)
                                if self.test_size > 0:
                                    self.X_train = self.X_train.reshape(-1,T,1)
                            self.x_shape = len(self.X.shape)

                        if self.x_shape == 2:
                            start = next = Input(shape=(T,))
                        elif self.x_shape > 2:
                            start = next = Input(shape=(T,1))
                    next = eval(v)(next)
                    i += 1
                model = tf.keras.models.Model(start, next)
            else:
                for v in df[c.VALUE]:
                    model.add(eval(v))

            h.debug(f'estimaor: {self.estimator}')
            eval(self.estimator)
            fit = self.model_func_df[self.model_func_df[c.KEY].str.startswith('fit')]
            h.debug(f"fit: {fit[c.VALUE].str.replace('fit','model.fit')}")

            if 'fit' in self.model_dict.keys():
                f = self.model_dict["fit"]
                p = f["params"]
                train = partial(model.fit, **p)

            if self.test_size:
                if self.has_y:
                    self.train = train(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test))
                else:
                    self.train = train(x=self.X_train, validation_data=(self.X_test))
            else:
                if self.has_y:
                    self.train = train(x=self.X, y=self.y)
                else:
                    self.train = train(x=self.X)

            if self.test_size:
                if self.has_y:
                    model.evaluate(self.X_test, self.y_test)
                else:
                    model.evaluate(self.X_test)
            else:
                if self.has_y:
                    model.evaluate(self.X, self.y)
                else:
                    model.evaluate(self.X)


            if self.has_y:
                n = len(np.unique(self.y_test)) if self.test_size else len(np.unique(self.y))
            elif self.model_dict['loss'].find('binary') != -1:
                n = 2
            else:
                n = -1

            if n == 2:
                self.y_pred = model.predict(self.X_test) if self.test_size else model.predict(self.X)
            elif self.data_type == c.LSTM:
                self.y_pred = model.predict(self.X_test) if self.test_size else model.predict(self.X)
            else:
                self.y_pred = model.predict(self.X_test).argmax(axis=1) if self.test_size else model.predict(self.X).argmax(axis=1)

            if self.data_type == c.LSTM and self.sc_X:
                self.y_pred = self.sc_X.inverse_transform(self.y_pred)

            h.debug(f'y_pred (after): {self.y_pred}')

            self.model = model
            self.target_continous = True

            if self.has_y:
                if isinstance(self.y, pd.DataFrame):
                    y = self.y.values
                else:
                    y = self.y
                if h.is_integer(y) and not h.is_integer(self.y_pred):
                    logger.info('y is discrete')
                    self.y_pred = (self.y_pred > 0.5)
                else:
                    logger.info('y is continous')
                    self.target_continous = True
                    if self.model_add and  self.timesteps > 1:
                        self.multistep_forecast()

            if self.save:
                self.save_xlsx(model)

    @h.measure_performance
    def multistep_forecast(self):
        """
        Function to forecast future predictions using the updated y-hat

        References
        ----------
        using yhat forecast - https://www.udemy.com/course/deep-learning-recurrent-neural-networks-in-python/learn/lecture/21514726#questions

        """
        if self.test_size <= 0:
            return

        validation_target = np.ravel(self.y_test.values) if isinstance(self.y_test, (pd.DataFrame, pd.Series)) else np.ravel(self.y_test)

        validation_predictions = []
        last_x = np.array(self.X_test.iloc[0,:].values) if isinstance(self.X_test, (pd.DataFrame, pd.Series)) else np.array(self.X_test[0])

        i = 0
        while len(validation_predictions) < len(validation_target):
            if self.modifier == c.ANN:
                p = self.model.predict(last_x.reshape(1,-1))[0,0]
            elif self.modifier == c.RNN:
                p = self.model.predict(last_x.reshape(1,-1,1))[0,0]
            validation_predictions.append(p)

            last_x = np.roll(last_x,-1)
            last_x[-1] = p
            i += 1

        self.validation_target = validation_target
        self.validation_predictions = validation_predictions

    @h.measure_performance
    def text_preprocess(self):
        """
        Function to do text preprocessing
        """


        X = self.X.squeeze()

        # split up the data
        df_train, df_test, self.y_train, self.y_test = train_test_split(X, self.y, test_size=self.prep.test_size, shuffle=False)

        # Convert sentences to sequences
        tokenizer = Tokenizer(num_words=c.MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(df_train)
        sequences_train = tokenizer.texts_to_sequences(df_train)
        sequences_test = tokenizer.texts_to_sequences(df_test)

        # get word -> integer mapping
        word2idx = tokenizer.word_index
        V = len(word2idx)
        logger.info('Found %s unique tokens.' % V)

        # pad sequences so that we get a N x T matrix
        self.X_train = pad_sequences(sequences_train)
        logger.info('Shape of data train tensor:', self.X_train.shape)

        # get sequence length
        T = self.X_train.shape[1]

        self.X_test = pad_sequences(sequences_test, maxlen=T)
        logger.info('Shape of data test tensor:', self.X_test.shape)

        # We get to choose embedding dimensionality
        D = 20

        logger.info(f'V: {V}, T: {T}, D: {D}')

        return [V, T, D]

    @h.measure_performance
    def save_xlsx(self, model):
        """
        Function to save the xlsx

        Parameters
        ----------
           x (dict): preprocessor method arguments

        References:
        ----------
           https://stackoverflow.com/questions/63843093/neural-network-summary-to-dataframe
        """


        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summ_string = "\n".join(stringlist)

        table = stringlist[1:-4][1::2]  # take every other element and remove appendix
        h.debug(table)

        n = len(stringlist)-2

        total = stringlist[-5::][1::1]
        total.pop()
        h.debug(f'total:{total}')

        # for the content
        new_table = []
        for entry in table:
            entry = re.split(r'\s{2,}', entry)[:-1]  # remove whitespace
            new_table.append(entry)

        df = pd.DataFrame(new_table[1:], columns=new_table[0])

        # for the appendix
        new_total = []
        for entry in total:
            entry = re.split(r':', entry)
            new_total.append(entry)

        new_total = np.array(new_total)

        #df1 = pd.DataFrame(data=list(new_total[:,1]), index=list(new_total[:,0]), columns=['Count'])
        df1 = pd.DataFrame({'label':list(new_total[:,0]), 'Count':list(new_total[:,1])})

        n = len(df) + 2

        if os.path.exists(self.xls_file):
            with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                df.to_excel(writer, sheet_name='input model')
                df1.to_excel(writer, sheet_name='input model', startrow=n)
        else:
            with pd.ExcelWriter(self.xls_file) as writer:
                df.to_excel(writer, sheet_name='input model')
                df1.to_excel(writer, sheet_name='input model', startrow=n)

        if self.has_y:
            if self.data_type == c.LSTM and self.sc_X:
                y_true = self.y_test_unscaled if self.test_size else self.y_unscaled
            else:
                y_true = self.y_test if self.test_size else self.y

            if isinstance(y_true, (pd.DataFrame, pd.Series)):
                y_true = np.ravel(y_true.values)

            self.y_true = y_true
            # added to safeguard
            if self.data_type == c.LSTM and self.y_pred.shape[1] > 1:
                self.y_pred = self.y_pred[:,0]


            if len(y_true) > 0 and (len(y_true) == len(self.y_pred)):
                if self.model_add and self.target_continous:
                    self.pd_y = pd_y = pd.DataFrame({self.ylabel: y_true, c.Y_PREDICT: self.validation_predictions})
                else:
                    self.pd_y = pd_y = pd.DataFrame({self.ylabel: y_true, c.Y_PREDICT :list(map(int,np.ravel(self.y_pred)))})

                with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                    pd_y.to_excel(writer, sheet_name='y-true and y-predict')

        # for local image handling of one image prediction
        h.debug(f'test_image: {self.prep.test_image}')
        if self.data_type == c.IMAGE:
            result = self.model.predict(self.prep.test_image)
            labels = self.prep.target
            i = int(result[0][0])
            df_image = pd.DataFrame({'File': [os.path.basename(self.prep.predict_file)], 'Predicted':[labels[i]]})
            with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                df_image.to_excel(writer, sheet_name='image prediction')

        self.save_stats()

    @h.measure_performance
    def save_stats(self):
        """
        Function to save the metrices like the accuracy, losses of the neural network
        """

        h.debug(self.train.history)

        dict = self.train.history
        n = len(list(self.train.history.values())[0])
        dict['epoch'] = range(1,n+1)
        df = pd.DataFrame(dict)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.plot_data = df = df[cols]

        if os.path.exists(self.xls_file):
            with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                df.to_excel(writer, sheet_name='stats')
        else:
            with pd.ExcelWriter(self.xls_file) as writer:
                df.to_excel(writer, sheet_name='stats')

        self.summary_stats()

    @h.measure_performance
    def plot_confusion_matrix(self):
        """
            Function to plot the confusion matrix
        """
        plt.clf()

        y = self.y if self.y_test is None else self.y_test

        if isinstance(y,pd.DataFrame):
            y = y.values

        cm = confusion_matrix(y, self.y_pred)
        i, j = cm.shape
        self.cm_df = pd.DataFrame(cm, columns=range(i), index=range(j))
        self.plot_cm()

        self.plot_confusion_example()

    @h.measure_performance
    def plot_confusion_example(self):
        """
            Function to plot the confusion example
        """
        if self.graph_dict:
            file = self.filename.split('.')[-1]
            labels = self.graph_dict[c.YLABEL]
            x, y = self.X_test[0].shape
            h.debug(f'x: {x}, y:{y}')
            misclassified_idx = np.where(self.y_pred != self.y_test)[0]
            i = np.random.choice(misclassified_idx)
            h.debug(f'y_test: {self.y_test[i]}, {labels[self.y_test[i]]}')
            h.debug(f'y_pred: {self.y_pred[i]}, {labels[self.y_pred[i]]}')
            plt.clf()
            plt.imshow(self.X_test[i].reshape(x, y), cmap='gray')
            plt.title(f"True label: {self.y_test[i]}-{labels[self.y_test[i]]}, Predicted: {self.y_pred[i]}-{labels[self.y_pred[i]]}")
            if self.graph:
                file2 = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + file + '_cm_misclassified_example_', c.DTTM_FMT,c.GRAPHIC_EXT)
                plt.savefig(file2)
            plt.show()

    @h.measure_performance
    def plot_3d(self):
        """
        Function to 3d plot the model of prediction surface
        """
        if not isinstance(self.X, pd.DataFrame):
            return

        if self.X.shape[1] < 2:
            return

        X = self.X.values
        y = self.y.values

        self.fig = plt.figure()
        ax = Axes3D(self.fig)
        ax.scatter(X[:, 0], X[:, 1], y)

        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_3d', c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)

    @h.measure_performance
    def plot_target(self):
        """
        Function to plot the y target: y-true value and y predicted

        References
        ----------
        pd.melt - https://pandas.pydata.org/docs/reference/api/pandas.melt.html
        """
        if not self.has_y:
           return

        if not hasattr(self, 'pd_y'):
            return

        pd_y = self.pd_y.copy()
        pd_y['index'] = list(self.pd_y.index + 1)

        pd_y = pd.melt(pd_y, id_vars=['index'], value_vars=[self.ylabel,c.Y_PREDICT])

        print(pd_y)

        self.fig, self.axs = plt.subplots()

        sns.lineplot(data=pd_y, x="index", y="value", hue="variable", ax=self.axs)
        if self.modifier_type:
            self.axs.set_title(f'{self.modifier_type} Lineplot: {self.filename.title()} ({self.ylabel} vs. {c.Y_PREDICT})')
        else:
            self.axs.set_title(f'Lineplot: {self.filename.title()} ({self.ylabel} vs. {c.Y_PREDICT})')
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_' + c.Y_PREDICT , c.DTTM_FMT, c.GRAPHIC_EXT)
            h.debug(f'file: {file}')
            self.fig.savefig(file)

    @h.measure_performance
    def plot(self):
        """
        Function to plot the model

        References:
        -----------
            lineplot - https://seaborn.pydata.org/generated/seaborn.lineplot.html
        """


        if self.has_y:
            #print(self.y.dtype)
            if isinstance(self.y, pd.DataFrame):
                y = self.y.values
            else:
                y = self.y

            if (h.is_integer(y)):
                logger.info(f'y is a discrete value')
                print('y is a discrete value')
                self.plot_confusion_matrix()
            else:
                print('y is a continous  value')
                logger.info(f'y must be a continous value')
                self.plot_3d()
                self.plot_target()

        n = len(self.neural_scores)
        h.debug(f'n: {n}')

        self.fig, self.axs = plt.subplots(n, 1)
        i = 0

        for key in self.train.history.keys():

            if list(filter(lambda score: key in score, self.neural_scores)):
                plot_data = pd.DataFrame({})
                plot_data['epoch'] = self.plot_data['epoch']
                plot_data[key] = self.plot_data[key]

                if self.test_size:
                    val_key = 'val_' + key
                    plot_data[val_key] = self.plot_data[val_key]

                print(plot_data)
                #exit(0)
                sns.lineplot('epoch', 'value', hue='variable',
                             data=pd.melt(plot_data, 'epoch'), ax=self.axs[i])

                if self.modifier_type:
                    self.axs[i].set_title(f'{self.modifier_type} Lineplot: {self.filename.title()} {key.title()} Scores')
                else:
                    self.axs[i].set_title(f'Lineplot: {self.filename.title()} {key.title()} Scores')
                i += 1


        for ax in self.axs.flat:
            ax.label_outer()

        plt.tight_layout()
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)

    def execute(self):
        """
        Function to execute the model providing plot
        """
        try:
            self.train_predict()
            self.plot()
        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)
