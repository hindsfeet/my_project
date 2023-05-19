"""
This is for regression/ classification model implementation

MCCT (initial)/ Minnie Cherry Chua Tan 10-Jul-21 Base version 2 coded from scratch
MCCT (initial)/ Minnie Cherry Chua Tan 21-Aug-21 Updated to train_predict, added the xls and the linear plot
MCCT (initial)/ Minnie Cherry Chua Tan 22-Aug-21 Updated plot_lr() to make the title and labels to autogenerate
MCCT (initial)/ Minnie Cherry Chua Tan 28-Aug-21 Updated for polynomial plot
MCCT (initial)/ Minnie Cherry Chua Tan 01-Sep-21 Added handling for SVR and plot
MCCT (initial)/ Minnie Cherry Chua Tan 02-Sep-21 Update to handle configurable predict_test in the input file
MCCT (initial)/ Minnie Cherry Chua Tan 04-Sep-21 Added handling for scaler and DecisionTreeRegressor, RandomForestRegressor
MCCT (initial)/ Minnie Cherry Chua Tan 06-Sep-21 Added classification: LogisticRegression, classification report
MCCT (initial)/ Minnie Cherry Chua Tan 07-Sep-21 Added plot_report(), plot_classification_scatter()
MCCT (initial)/ Minnie Cherry Chua Tan 10-Sep-21 Added set_label(), updated plot_classification_scatter()
MCCT (initial)/ Minnie Cherry Chua Tan 13-Sep-21 Added plot_pairplot()
MCCT (initial)/ Minnie Cherry Chua Tan 16-Sep-21 Updated the ypred_df to have X values in the excel,
    updated plot_classification_scatter() error handling and updated scaling in plot_contours(),
MCCT (initial)/ Minnie Cherry Chua Tan 16-Sep-21 Refactor to model.py - report_classification_cm(), set_label(),
    plot_report(), plot_pairplot()
MCCT (initial)/ Minnie Cherry Chua Tan 31-Mar-22 Updated plot_scatter() for seaborn as Python ax.plot in axis has been change in Python binary
    to_values() is added with data format changed locally with switch server.
MCCT (initial)/ Minnie Cherry Chua Tan 01-Apr-22 Updated train_predict() to have np.newaxis as default python library has been switched
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student

__author__ = 'Minnie Tan'
#Chua is my middle name so I somtimes omit as per my mother's surname (Melody Chua), Tan is my father's english surname (Julio Tan)

from model import Model
from sklearn.preprocessing import *

# for classification
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

from constant import Constant as c
from helper import Helper as h

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class RegressionClassification(Model):
    @h.measure_performance
    def train_predict(self):
        """
        Function to train and predict the model using the training sets
        """
        print(f'train and predict: {self.estimator}')

        print(f'{type(self.estimator)}, {isinstance(self.estimator,str)}')
        print(f'test_size: {self.test_size}')
        print(f'sc_y: {self.sc_y}')
        print(f'sc_X: {self.sc_X}')

        if self.test_size:
            print(f'train  {self.X_train} {self.y_train}')
            print(f'test  {self.X_test} {self.y_test}')

            print(f'shape  {self.X_train.shape} {self.y_train.shape}')
            print(f'shape  {self.X_test.shape} {self.y_test.shape}')

        self.classifier = self.estimator

        print(self.test_size)
        if self.test_size:
            print(f'test_size: {self.test_size}')
            self.classifier.fit(self.X_train, self.y_train)

            print(f'X_train: {self.X_train}, y_train: {self.y_train}')

            #self.y_pred = self.classifier.predict(self.X_test)
            #this is for test split
            if self.sc_y is None:
                print(type(self.X_test))
                print(self.X_test.shape)
                print(self.X_test)
                self.y_pred = self.classifier.predict(self.X_test)
                print(f'y_pred: {self.y_pred}')
            else:
                print('scaler1')

                self.y_pred = self.sc_y.inverse_transform(self.classifier.predict(self.X_test))
                #self.y_test = np.ravel(self.sc_y.inverse_transform(self.y_test))
                self.y_test = np.ravel(self.y_test_unscaled)
            # else:
            #     self.y_pred = self.classifier.predict(self.X_test)

            print(f'y_pred: {self.y_pred}, y_test: {self.y_test}')
            ypred_df = pd.DataFrame({'y_pred': self.y_pred, 'y': self.y_test})
            print(ypred_df)
        else:
            print(f'no test_size')
            print(f'scale X: {self.X}, y:{self.y}')
            self.classifier.fit(self.X, self.y)
            print(f'self.sc_y: {self.sc_y}')

            if self.sc_y is None:
                print('has no scaler')
                self.y_pred = self.classifier.predict(self.X)
                ypred_df = pd.DataFrame({'y_pred':self.y_pred, 'y': self.y})
                print(ypred_df)
            else:
                print('has scaler')
                print(f'unscaled: X: {self.X_unscaled} y:{self.y_unscaled}')

                print(type(self.X))
                y_pred = self.classifier.predict(self.X)
                y_pred = y_pred[:, np.newaxis]
                print(f'y_pred: {y_pred.shape}')
                self.y_pred = self.sc_y.inverse_transform(y_pred)
                print(f'y_pred: {self.y_pred.shape}')

                self.y = np.ravel(self.sc_y.inverse_transform(self.y))
                print(f'self.y: {self.y.shape}')
                self.y = self.y[:, np.newaxis]

                print(f'y_pred: {self.y_pred}, y:{self.y}')

                ypred_df = pd.DataFrame({'y_pred': np.ravel(self.y_pred), 'y': np.ravel(self.y)})
                print(ypred_df)

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
        print(ypred_df)

        print(self.predict_test)

        # this is for user input in input*.txt
        if self.predict_test is not None:
            print(f'predict_test: {type(self.predict_test)}')
            self.predict_test = np.ravel(self.predict_test)
            self.predict_test = self.predict_test[:, np.newaxis]
            print(self.sc_X)
            X_test = self.sc_X.transform(self.predict_test) if self.sc_X else self.predict_test
            print(X_test)
            self.y_predict_test = self.classifier.predict(X_test)
            #self.y_predict_test = self.classifier.predict(self.predict_xtest)
            self.y_predict_test = self.y_predict_test[:, np.newaxis]
            print(self.sc_y)
            self.y_predict_test = self.sc_y.inverse_transform(self.y_predict_test) if self.sc_y else self.y_predict_test
            print(f'y_predict_test: {self.y_predict_test} {self.ylabel}')
            label = [",".join(self.xlabel), c.PREDICT_TEST]
            print(type(self.predict_test))
            print(label)
            self.y_predict_test = np.ravel(self.y_predict_test)
            self.y_predict_test = self.y_predict_test[:, np.newaxis]
            predict_test_df = pd.DataFrame(np.vstack([self.predict_test, self.y_predict_test]), index=label).T
            print(predict_test_df)

        if self.save:
            print('save')
            self.summary_stats()

            if os.path.exists(self.xls_file):
                with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                    ypred_df.to_excel(writer, sheet_name='y-pred')
            else:
                with pd.ExcelWriter(self.xls_file) as writer:
                    ypred_df.to_excel(writer, sheet_name='y-pred')

            print(self.y_pred.dtype)

            if self.predict_test is not None:
                with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                    predict_test_df.to_excel(writer, sheet_name=c.PREDICT_TEST)

            self.report_classification_cm()

    @h.measure_performance
    def plot_scatter(self, type_set: str, ax: plt.Axes):
        """
        Function to plot the scatter graph and line for linear regression

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants or NONE if no test_size
           ax (plt.Axes): specify axes of the plot
        """
        print(f'plot_scatter: {type_set}')
        print(f'{type(self.X)} {type(self.y)}')
        print(c.XLABEL in self.graph_dict)

        # scale back for graphing
        if self.test_size is not None:
            print(f'X_train: {self.X_train} X_test: {self.X_test}')
            print(f'unscaled: X_train: {self.X_train_unscaled} X_test: {self.X_test_unscaled} ')

            if self.sc_y is None:
                y_train = self.classifier.predict(self.X_train)
            else:
                y_train = self.sc_y.inverse_transform(self.classifier.predict(self.X_train))
                self.X_train = self.X_train_unscaled
                self.X_test = self.X_test_unscaled

            if self.smooth > 0:
                print(f"smooth: {self.smooth}")
                if self.sc_X is None:
                    X_grid = np.arange(min(self.X_train.values), max(self.X_train.values), self.smooth)
                else:
                    print(f'min: {min(self.X_train_unscaled.values)}, max: {max(self.X_train_unscaled.values)}')
                    X_grid = np.arange(min(self.X_train_unscaled.values), max(self.X_train_unscaled.values), self.smooth)
                print(X_grid.shape)
                X_grid = X_grid[:, np.newaxis]
                print(X_grid.shape)
                if not self.has_scaler:
                    y_train = self.classifier.predict(X_grid)
                else:
                    y_train = self.sc_y.inverse_transform(self.classifier.predict(self.sc_X.transform(X_grid)))
        else:

            if self.sc_X is not None:
                self.X = self.X_unscaled
            if self.smooth > 0:
                print(f"smooth: {self.smooth}")
                if self.sc_X is None:
                    X_grid = np.arange(min(self.X.values), max(self.X.values), self.smooth)
                else:
                    print(f'min: {min(self.X_unscaled.values)}, max: {max(self.X_unscaled.values)}')
                    X_grid = np.arange(min(self.X_unscaled.values), max(self.X_unscaled.values), self.smooth)

                print(X_grid.shape)
                #X_grid = X_grid.reshape((len(X_grid),1))
                X_grid = X_grid[:, np.newaxis]
                print(X_grid.shape)
                #print(self.sc_X.transform(X_grid))
                if not self.has_scaler:
                    print('has no scaler')
                    self.y_pred = self.classifier.predict(X_grid)
                else:
                    y_pred = self.classifier.predict(self.sc_X.transform(X_grid))
                    y_pred = y_pred[:, np.newaxis]
                    self.y_pred = self.sc_y.inverse_transform(y_pred)
                    print(self.y_pred.shape)

                print(f'y_pred: {self.y_pred}')

        if c.TRAINING_SET == type_set:
            print(f'X_train: {self.X_train}, y_train: {self.y_train}')
            sns.scatterplot(x=h.to_values(self.X_train), y=h.to_values(self.y_train), ax=ax)
            title = self.title + ' (' + c.TRAINING_SET + ')' if self.title else self.modifier + '(' + c.TRAINING_SET + ')'
        elif c.TEST_SET == type_set:
            print(f'X_test: {self.X_test}, y_test: {self.y_test}')
            sns.scatterplot(x=h.to_values(self.X_test), y=h.to_values(self.y_test), ax=ax)
            title = '(' + c.TEST_SET + ')'
        else:
            print(f'X: {self.X}, y: {self.y}')
            sns.scatterplot(x=h.to_values(self.X), y=h.to_values(self.y), ax=ax)
            title = self.title

        if self.smooth > 0:
            if type_set in [c.TRAINING_SET, c.TEST_SET]:
                sns.lineplot(x=h.to_values(X_grid), y=h.to_values(y_train), ax=ax, color='red')
            else:
                if self.modifier == 'PolynomialFeatures':
                    print(f'X: {self.X}, y: {self.y}')
                    sns.lineplot(x=h.to_values(self.X), y=h.to_values(self.y), ax=ax)
                    sns.lineplot(x=h.to_values(X_grid), y=h.to_values(self.y_pred), ax=ax, color='red')
                    ax.legend(loc="best")
                else:
                    sns.lineplot(x=h.to_values(X_grid), y=h.to_values(self.y_pred), ax=ax, color='red')
        else:
            if type_set in [c.TRAINING_SET, c.TEST_SET]:
                sns.lineplot(x=h.to_values(self.X_train), y=h.to_values(y_train), ax=ax, color='red')


        self.set_label(type_set, ax)


    def plot_contours(self, ax: plt.Axes, xx: np.array, yy: np.array, **params: object):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax (plt.Axes): matplotlib axes object
        xx (np.array): meshgrid ndarray
        yy (np.array): meshgrid ndarray
        params (object): dictionary of params to pass to contourf, optional

        Return
        ------
        contour plot
        """

        print(f'smooth: {self.smooth}')
        if len(self.smooth) == 2:
            #X_grid = np.array([X1.ravel(), X2.ravel()]).T
            X_grid = np.c_[xx.ravel(), yy.ravel()]
            print(X_grid)
            if self.sc_X is None:
                Z = self.classifier.predict(X_grid)
            else:
                if self.ext != c.CSV:
                    sc_X = eval(self.scaler)
                    Z = self.classifier.predict(sc_X.fit_transform(X_grid))
                else:
                    Z = self.classifier.predict(self.sc_X.transform(X_grid))
            if self.sc_y is not None:
                if ext == c.CSV:
                    sc_y = eval(self.scaler)
                    Z = sc_y.inverse_transform(Z)
                else:
                    Z = self.sc_y.inverse_transform(Z)
            Z = Z.reshape(xx.shape)

            return ax.contourf(xx, yy, Z, **params)

    @h.measure_performance
    def plot_classification_scatter(self, type_set: str, ax: plt.Axes):
        """
        Function to plot the scatter graph and contour for classification

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants
           ax (plt.Axes): specify axes of the plot

        Reference
        ---------
            colormap - https://matplotlib.org/stable/tutorials/colors/colormaps.html
            contour - https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
            legend properties - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        """
        print(f'plot_scatter: {type_set}')
        print(f'test_size: {self.test_size} {type(self.X)} {type(self.y)} ')

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
            print(f'self.smooth: {self.smooth} {self.smooth[0]} {len(self.smooth)}')

            X = self.X_train if type_set == c.TRAINING_SET else self.X_test
            y = self.y_train if type_set == c.TRAINING_SET else self.y_test

            if len(self.smooth) == 2:
                h1, h2 = self.smooth[0], self.smooth[1]
                print(f'h1: {h1}, h2: {h2} {type(self.X_train)}')
                if type_set == c.TRAINING_SET:
                    X1, X2 = np.meshgrid(np.arange(start=self.X_train.iloc[:, 0].min() - 1, stop=self.X_train.iloc[:, 0].max() + 1, step=h1),
                        np.arange(start=self.X_train.iloc[:, 1].min() - 1, stop=self.X_train.iloc[:, 1].max() + 1, step=h2))
                elif type_set == c.TEST_SET:
                    X1, X2 = np.meshgrid(np.arange(start=self.X_test.iloc[:, 0].min() - h1, stop=self.X_test.iloc[:, 0].max() + h1,step=h1),
                        np.arange(start=self.X_test.iloc[:, 1].min() - 1, stop=self.X_test.iloc[:, 1].max() + 1,step=h2))

                print(f'X1: {X1}, X2: {X2}')
                print(f'x1: {X1.size}, x2:{X2.size}')

        else:

            if self.sc_X is not None:
                self.X = self.X_unscaled
            if self.sc_y is not None:
                self.y = self.y_unscaled
            X = self.X
            y = self.y
            print(f'self.smooth: {self.smooth}')

            if len(self.smooth) == 2:
                h1, h2 = self.smooth[0], self.smooth[1]
                print(f'h1: {h1}, h2: {h2} {type(self.X)}')
                X1, X2 = np.meshgrid(np.arange(start=self.X.iloc[:, 0].min() - 1, stop=self.X.iloc[:, 0].max() + 1, step=h1),
                                     np.arange(start=self.X.iloc[:, 1].min() - 1, stop=self.X.iloc[:, 1].max() + 1, step=h2))
                print(f'X1: {X1}, X2: {X2}')
                print(f'x1: {X1.size}, x2:{X2.size}')

        self.plot_contours(ax, X1, X2, cmap=plt.cm.coolwarm, alpha=0.8)

        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.legend(*scatter.legend_elements(), loc='best', title=self.y.name, title_fontsize='x-small', frameon=True)
        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())
        self.ylabel = X.iloc[:, 1].name
        print(f'ylabel: {self.ylabel}')
        self.set_label(type_set, ax)


    @h.measure_performance
    def plot_lr(self):
        """
        Function to plot the linear model using scatter plot
        """
        h.debug(f'plot_lr: {self.modifier}, {self.test_size}')

        if self.test_size:
            print(f'test_size: {self.test_size}')

            self.fig, self.axs = plt.subplots(1, 2)

            self.plot_scatter(c.TRAINING_SET, self.axs[0])
            self.plot_scatter(c.TEST_SET, self.axs[1])

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in self.axs.flat:
                ax.label_outer()
            plt.show()
        else:
            self.fig, self.axs = plt.subplots(1, 1)
            self.plot_scatter(None, self.axs)

        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)


    @h.measure_performance
    def plot_classification(self):
        """
        Function to plot the 2D classification model using scatter plot
        """
        h.debug(f'plot_lr: {self.modifier}, {self.test_size}')
        # print(f'X: {self.X} y: {self.y}')

        _, i = self.X.shape if self.test_size is None else self.X_train.shape
        if i != 2:
            return

        if self.test_size is not None:
            print(f'test_size: {self.test_size}')

            self.fig, self.axs = plt.subplots(1, 2)
            self.plot_classification_scatter(c.TRAINING_SET, self.axs[0])
            self.plot_classification_scatter(c.TEST_SET, self.axs[1])
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in self.axs.flat:
                ax.label_outer()

        else:
            self.fig, self.ax = plt.subplots(1, 1)
            self.plot_classification_scatter(None, self.ax)

        plt.suptitle('Scatter Plot')
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)
            plt.show()


    @h.measure_performance
    def plot(self):
        """
        Function to plot the linear model using scatter plot
        """
        print('plot')
        print(f'shape x: {self.X.shape}')
        print(f'type: {self.estimator_type}')

        if self.estimator_type == c.REGRESSOR:
            self.plot_lr()
        elif self.estimator_type == c.CLASSIFIER:
            h.debug('plot classifier')
            if self.has_y:
                self.plot_pairplot('None')
                if self.test_size:
                    self.plot_pairplot(c.TRAINING_SET)
                    self.plot_pairplot(c.TEST_SET)

            if self.yvalue:
                self.plot_report(c.CLASSIFICATION_REPORT)
                self.plot_report(c.CONFUSION_MATRIX)
                self.plot_report(c.PREDICTION_ERROR)
                if self.modifier != 'SVC':
                    self.plot_report(c.ROC_AUC)
            self.plot_classification()



    def execute(self):
        """
        Function to execute the regression / classification providing plot
        """
        try:

            self.train_predict()
            self.plot()
        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)