"""
This is for unsupervised learning implementation

MCCT (initial)/ Minnie Cherry Chua Tan 2-Apr-22 Added template of train_predict()
MCCT (initial)/ Minnie Cherry Chua Tan 6-Apr-22 Updated train_predict() to have summary_stats(), added plot_cm()
MCCT (initial)/ Minnie Cherry Chua Tan 7-Apr-22 Added plot_som()
MCCT (initial)/ Minnie Cherry Chua Tan 8-Apr-22 Added plot_som_scatter(), updated plot_som()
MCCT (initial)/ Minnie Cherry Chua Tan 16-Apr-22 Added convert(), rate_data(), evaluate()
MCCT (initial)/ Minnie Cherry Chua Tan 21-Apr-22 Added plot_rbm(), updated evaluate() - add history of loss and accuracy score to xls
MCCT (initial)/ Minnie Cherry Chua Tan 22-Apr-22 Updated rate_data() to make the scoring setup generic
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

from model import Model
from constant import Constant as c
from helper import Helper as h
from sklearn_som.som import SOM
from sklearn import datasets
from rbm import RBM

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
import math
import torch

logger = logging.getLogger(__name__)

class Nonsupervised(Model):
    def convert(self, data: np.array, max_x: int, max_y: int) -> [[]]:
        """
        To convert the dataset

        Parameters
        ----------
            data (np.array): the dataset to be converted
            max_x (int): the maximum value for X
            max_y (int): the maximum value for y

        References
        ----------
        convert() - https://www.udemy.com/course/deeplearning/learn/lecture/6895658#overview
        """
        new_data = []
        for id_X in range(1, max_x + 1):
            id_y = data[:, 1][data[:, 0] == id_X]
            id_values = data[:, 2][data[:, 0] == id_X]
            values = np.zeros(max_y)
            values[id_y - 1] = id_values
            new_data.append(list(values))
        return new_data

    @h.measure_performance
    def rate_data(self, data: [[]], level: int, rate: int = 1) -> [[]]:
        """
        To convert the dataset proportionally according to the rate scores

        Parameters
        ----------
            data ([[]]): the dataset to be converted based on rate
            level (int): maximum value of the criterion used
            rate (int): rate scores including the 0

        Return
        ------
            new_data ([[]]): the coverted dataset based on the scores

        References
        ----------
        rates - https://www.udemy.com/course/deeplearning/learn/lecture/6895662#overview
        """

        data = np.array(data)

        new_data = np.zeros(data.shape)
        new_data[data == 0] = -1
        new_data[data > 0] = 0
        if level == 0 or rate == 0:
            return

        assert (rate + 1) <= level, 'Error: rate (' + rate + 1 + ') cannot exceed maximum number (' + level + ') in the dataset'

        score = rate
        quotient = int(level / (rate + 1))
        upper = math.ceil(level / (rate + 1))
        lower = math.floor(level / (rate + 1))
        remainder = level % (rate + 1)

        to = level
        counter = remainder + 1
        while to > 0:
            if counter == 0:
                step = lower
            else:
                step = upper
                counter -= 1
                if counter == 0:
                    step = lower
            i = to - step + 1

            new_data[(data >= i) & (data <= to)] = score

            to -= step
            score -= 1

        return torch.FloatTensor(new_data)

    @h.measure_performance
    def process_files(self):
        """
        To preprocess file by name
        """
        if self.filename == 'movies':
            self.X_train = np.array(self.X_train, dtype='int')
            self.X_test = np.array(self.X_test, dtype='int')

            nb_users = int(max(max(self.X_train[:, 0], ), max(self.X_test[:, 0])))
            nb_movies = int(max(max(self.X_train[:, 1], ), max(self.X_test[:, 1])))
            nb_ratings = int(max(max(self.X_train[:, 2], ), max(self.X_test[:, 2])))


            self.X_train = self.convert(self.X_train, nb_users, nb_movies)
            self.X_test = self.convert(self.X_test, nb_users, nb_movies)

            self.X_train = self.rate_data(self.X_train, nb_ratings)
            self.X_test = self.rate_data(self.X_test, nb_ratings)
            self.n = nb_users

    @h.measure_performance
    def evaluate(self):
        """
        To evaluate the test_set given the training set based on the hidden nodes, epoch and batch_size using Gibb Sampling

        Reference:
        https://www.udemy.com/course/deeplearning/learn/lecture/6895698#overview
        https://www.udemy.com/course/deeplearning/learn/lecture/6895700#overview
        https://www.udemy.com/course/deeplearning/learn/lecture/6895706#overview
        """
        n = self.n
        batch_size = self.batch_size
        epoch = self.epoch

        self.history = history = pd.DataFrame(columns=[c.EPOCH,c.LOSS,c.ACCURACY])

        logger.info(f'n: {n}, batch_size: {batch_size}, epoch: {epoch}')

        for epoch in range(1, epoch + 1):
            accuracy = train_accuracy = train_loss = 0

            s = 0.
            for id_user in range(0, n - batch_size, batch_size):
                vk = self.X_train[id_user: id_user + batch_size]
                v0 = self.X_train[id_user: id_user + batch_size]
                _, ph0 = self.rbm.sample_h(v0)
                for k in range(10):
                    _, hk = self.rbm.sample_h(vk)
                    _, vk = self.rbm.sample_v(hk)
                    vk[v0 < 0] = v0[v0 < 0]
                phk, _ = self.rbm.sample_h(vk)

                self.rbm.train(v0, vk, ph0, phk)
                # mae - mean average error - distance
                train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
                train_accuracy += sum(v0[v0 >= 0] == vk[v0 >= 0]) / float(len(v0[v0 >= 0]))  # -> you get 0.75

                # root mean square error
                #train_rmse += np.sqrt(torch.mean((v0[v0 >= 0] - vk[v0 >= 0]) ** 2))
                s += 1.
            accuracy += 1 - (train_loss / s)
            history.loc[epoch - 1, 'epoch'] = epoch
            history.loc[epoch -1,'loss'] = float(train_loss / s)
            history.loc[epoch -1, 'accuracy'] = float(train_accuracy / s)
            print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s) + ', train_accuracy: ' + str(
                train_accuracy / s) + ', accuracy:' + str(accuracy))


        """## Testing the RBM"""
        history_test = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
        accuracy = test_accuracy = test_loss = 0
        s = 0.
        for id_user in range(n):
            v = self.X_train[id_user:id_user + 1]
            vt = self.X_test[id_user:id_user + 1]
            if len(vt[vt >= 0]) > 0:
                _, h = self.rbm.sample_h(v)
                _, v = self.rbm.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
                test_accuracy += sum(vt[vt >= 0] == v[vt >= 0]) / float(len(vt[vt >= 0]))
                # test_loss += np.sqrt(torch.mean((vt[vt >= 0] - v[vt >= 0]) ** 2))
                s += 1.

        history_test.loc[epoch - 1, 'epoch'] = 1
        history_test.loc[epoch - 1, 'loss'] = float(test_loss / s)
        history_test.loc[epoch - 1, 'accuracy'] = float(test_accuracy / s)

        print('test loss: ' + str(test_loss / s))
        accuracy = 1 - (test_loss / s)
        # print('test accuracy: '+str(accuracy/s))
        print('test loss: ' + str(test_loss / s) + ' test_accuracy:' + str(test_accuracy / s) + ', accuracy:' + str(accuracy))

        if os.path.exists(self.xls_file):
            with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                history.to_excel(writer, sheet_name='stats')
                history_test.to_excel(writer, sheet_name='stats_test')
        else:
            with pd.ExcelWriter(self.xls_file) as writer:
                history.to_excel(writer, sheet_name='stats')
                history_test.to_excel(writer, sheet_name='stats_test')

    @h.measure_performance
    def train_predict(self):
        """
        Function to train and predict the model using the datasets
        """
        h.debug(f'train and predict: {self.modifier}')

        if self.modifier == c.RBM:
            if self.filename == 'movies':
                self.process_files()
            visible = len(self.X_train[0])
            hidden = self.hidden
            self.rbm = RBM(visible, hidden)
            self.evaluate()
        elif self.modifier == c.SOM:
            m = len(np.unique(self.y)) if self.has_y else None
            if not m:
                return
            n = 1  # ??? mxn neurons
            dim = self.X.shape[1]
            self.classifier = self.modifier + '(m=' + str(m) + ', n=' + str(n) + ', dim=' + str(dim) + ')'
            self.classifier = eval(self.classifier)
        else:
            self.classifier = eval(self.estimator) if isinstance(self.estimator,str) else self.estimator

        if self.modifier != c.RBM:
            if self.test_size:
                h.debug(f'test_size: {self.test_size}')
                if self.modifier == c.SOM:
                    if isinstance(self.X_train, (pd.DataFrame, pd.Series)):
                        X_train = self.X_train.values
                    self.classifier.fit(X_train)
                else:
                    self.classifier.fit(self.X_train, self.y_train)
                if self.sc_y is None:
                    if self.modifier == c.SOM:
                        if isinstance(self.X_test, (pd.DataFrame, pd.Series)):
                            X_test = self.X_test.values
                        self.y_pred = self.classifier.predict(X_test)
                    else:
                        self.y_pred = self.classifier.predict(self.X_test)
                else:

                    self.y_pred = self.sc_y.inverse_transform(self.classifier.predict(self.X_test))
                    self.y_test = np.ravel(self.y_test_unscaled)

                ypred_df = pd.DataFrame({'y_pred': self.y_pred, 'y': self.y_test})
            else:
                if self.modifier == c.SOM:
                    if isinstance(self.X, (pd.DataFrame, pd.Series)):
                        X = self.X.values
                    self.classifier.fit(X)
                else:
                    self.classifier.fit(self.X, self.y)

                if self.sc_y is None:
                    if self.modifier == c.SOM:
                        self.y_pred = self.classifier.predict(X)
                    else:
                        self.y_pred = self.classifier.predict(self.X)
                    ypred_df = pd.DataFrame({'y_pred':self.y_pred, 'y': self.y})
                else:
                    if self.modifier == c.SOM:
                        y_pred = self.classifier.predict(X)
                    else:
                        y_pred = self.classifier.predict(self.X)
                    y_pred = y_pred[:, np.newaxis]
                    self.y_pred = self.sc_y.inverse_transform(y_pred)

                    self.y = np.ravel(self.sc_y.inverse_transform(self.y))
                    self.y = self.y[:, np.newaxis]


                    ypred_df = pd.DataFrame({'y_pred': np.ravel(self.y_pred), 'y': np.ravel(self.y)})

            if self.sc_X is not None:
                if self.X_unscaled is not None:
                    X_df = self.X_unscaled.copy()
                elif self.X_test_unscaled is not None:
                    X_df = self.X_test_unscaled.copy()
                ypred_df = ypred_df.join(X_df)
            else:
                if self.X is not None:
                    X_df = self.X.copy()
                elif self.X_test is not None:
                    X_df = self.X_test.copy()
                ypred_df = ypred_df.join(X_df)

            if self.save:
                print('save')
                self.summary_stats()
                if os.path.exists(self.xls_file):
                    with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                        ypred_df.to_excel(writer, sheet_name='y-pred')
                else:
                    with pd.ExcelWriter(self.xls_file) as writer:
                        ypred_df.to_excel(writer, sheet_name='y-pred')

                self.report_classification_cm()

    @h.measure_performance
    def plot_som(self, type_set: str):
        """
            Function to plot the SOM

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants or NONE if no test_size
        """
        fig, ax = plt.subplots(2,1)

        print(f'X: {self.X} X: {type(self.X)}')
        print(f'y: {self.y}, y_pred: {self.y_pred}')

        X = y = None

        if self.test_size is not None:
            print(len(self.X_train))

            print(f'X_train: {self.X_train} X_test: {self.X_test}')
            print(f'unscaled: X_train: {self.X_train_unscaled} X_test: {self.X_test_unscaled} ')

            if self.sc_y is not None:
                self.y_train = self.y_train_unscaled
                self.y_test = self.y_test_unscaled

            if self.sc_X is not None:
                self.X_train = self.X_train_unscaled
                self.X_test = self.X_test_unscaled

            X = self.X_train if type_set == c.TRAINING_SET else self.X_test
            y = self.y_train if type_set == c.TRAINING_SET else self.y_test
            print(f'shape: {X.shape} {y.shape} {self.y_pred.shape}')
        else:

            if self.sc_X is not None:
                self.X = self.X_unscaled
            if self.sc_y is not None:
                self.y = self.y_unscaled
            X = self.X
            y = self.y

        print(f'type X: {type(X)}, y: {type(y)}')

        ax[0].scatter(h.to_values(X.iloc[:, 0]), h.to_values(X.iloc[:, 1]), c=h.to_values(y), cmap=plt.cm.coolwarm)
        ax[0].title.set_text('Actual Classes')
        ax[1].scatter(h.to_values(X.iloc[:, 0]), h.to_values(X.iloc[:, 1]), c=h.to_values(self.y_pred), cmap=plt.cm.coolwarm)
        ax[1].title.set_text('SOM Predictions')

        plt.tight_layout()

        if self.graph:
            file = self.filename.split('.')[-1]
            # print(f'file: {file}')
            if self.test_size:
                file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + file + '_som_' + type_set + '_', c.DTTM_FMT, c.GRAPHIC_EXT)
            else:
                file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + file + '_som_', c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            plt.savefig(file)

    @h.measure_performance
    def plot_som_scatter(self):
        """
        Function to plot the linear model using scatter plot
        """
        h.debug(f'plot_som_scatter: {self.modifier}, {self.test_size}')

        if self.test_size:
            print(f'test_size: {self.test_size}')
            self.plot_som(c.TEST_SET)
        else:
            self.plot_som(None)

        plt.show()

    @h.measure_performance
    def plot_rbm(self):

        print(f'self.history (before): {self.history}')
        self.history = pd.melt(self.history, id_vars=[c.EPOCH], value_vars=[c.LOSS, c.ACCURACY])
        print(f'self.history (after): {self.history}')

        self.fig, self.axs = plt.subplots()

        sns.lineplot(data=self.history, x=c.EPOCH, y="value", hue="variable", ax=self.axs)
        self.axs.set_title(f'Lineplot: {self.filename.title()} ({c.LOSS} vs. {c.ACCURACY})')
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_' + c.LOSS + '_' + c.ACCURACY, c.DTTM_FMT,
                                                                    c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)

    @h.measure_performance
    def plot(self):
        """
        Function to plot the model
        """
        print(f'plot:')
        self.plot_cm()
        if self.modifier == c.SOM:
            self.plot_som_scatter()
        elif self.modifier == c.RBM:
            self.plot_rbm()
        plt.show()

    @h.measure_performance
    def execute(self):
        """
        Function to execute the rules providing plot
        """
        try:
            self.train_predict()
            self.plot()
        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)
