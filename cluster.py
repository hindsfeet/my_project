"""
This is for cluster model implementation

MCCT (initial)/ Minnie Cherry Chua Tan 15-Sep-21 Base skeleton version 2 coded from scratch
MCCT (initial)/ Minnie Cherry Chua Tan 19-Sep-21 Added the content in train_predict()
MCCT (initial)/ Minnie Cherry Chua Tan 20-Sep-21 Added the content in plot() - add plot_cluster(),
    plot_cluster_scatter(), plot_csm(), plot_silhouette()
MCCT (initial)/ Minnie Cherry Chua Tan 21-Sep-21 Added the plot_pairplot(), updated train_predict() to
    handle dataset with target using GridSearchCV, while without target uses Elbow method
MCCT (initial)/ Minnie Cherry Chua Tan 22-Sep-21 Updated plot_silhouette() to remove plot_csm and use Visualizer instead
MCCT (initial)/ Minnie Cherry Chua Tan 02-Oct-21 Added the plot_dendograph() for hierachical clustering
MCCT (initial)/ Minnie Cherry Chua Tan 03-Oct-21 Updated checking for the n_cluster if tuned or not, refactor plot_elbow()
MCCT (initial)/ Minnie Cherry Chua Tan 09-Oct-21 Added handling for DBSCAN
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

from model import Model
from constant import Constant as c
from helper import Helper as h
from yellowbrick.cluster import *
from sklearn.cluster import *
from sklearn.metrics import silhouette_samples

import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import logging
import os
from yellowbrick.cluster import silhouette_visualizer

logger = logging.getLogger(__name__)

class Cluster(Model):
    @h.measure_performance
    def train_predict(self):
        """
        Function to train and predict the model using the training sets
        """
        logger.info(f'train and predict: {self.estimator}')

        # models with n_clusters arguments tested
        if self.modifier not in ['DBSCAN', 'MeanShift', 'OPTICS']:
            k = len(np.unique(self.y)) if h.is_integer(self.y) and self.has_y else range(2,11)
            h.debug(f'k: {k}')
            self.no_n_cluster = False
        else:
            self.no_n_cluster = True
        if not self.has_y:
            self.plot_elbow(k)
            logger.info(f'cluster: {self.n_cluster}')
            self.estimator = eval(self.modifier + '( n_clusters=' + str(self.n_cluster) + ')')
            self.y_pred = self.estimator.fit_predict(self.X_test) if self.test_size else self.estimator.fit_predict(self.X)
            logger.info(f'y_pred: {self.y_pred}')

            if self.sc_y is not None:
                self.y_pred = self.sc_y.inverse_transform(self.y_pred)
            if self.sc_X:
                X = self.X_test_unscaled if self.test_size else self.X_unscaled
            else:
                X = self.X_test if self.test_size else self.X
            labels = ['y_pred']
            labels.extend(self.xlabel)
            ypred_df = pd.DataFrame(np.c_[self.y_pred, X.values], columns=labels)
            logger.info(ypred_df)

        else:
            if self.tuned or self.no_n_cluster:
                self.classifier = eval(str(self.estimator))
            else:
                if isinstance(k, int):
                    self.n_cluster = k
                else:
                    self.plot_elbow(k)
                self.classifier = self.estimator = eval(self.modifier + '( n_clusters=' + str(self.n_cluster) + ')')

            if self.test_size:
                # if not self.tuned:
                self.classifier.fit(self.X_train, self.y_train)
                if self.modifier != 'DBSCAN':
                    self.y_pred = self.classifier.predict(self.X_test)
                else:
                    if self.has_y:
                        self.y_pred = self.classifier.fit_predict(self.X_test, self.y_test)
                    else:
                        self.y_pred = self.classifier.fit_predict(self.X_test)


            else:
                if self.modifier != 'DBSCAN':
                    self.classifier.fit(self.X, self.y)
                    self.y_pred = self.classifier.predict(self.X)
                else:
                    if self.has_y:
                        self.y_pred = self.classifier.fit_predict(self.X, self.y)
                    else:
                        self.y_pred = self.classifier.fit_predict(self.X)


            if self.sc_y:
                logger.info('scale')
                self.y_pred = self.classifier.inverse_transform(self.y_pred)
                y = self.y_test_unscaled if self.test_size else self.y_unscaled
            else:
                y = self.y_test if self.test_size else self.y

            logger.info(f'y_pred: {self.y_pred} y:{y.values}')

            if self.sc_X:
                X = self.X_test_unscaled if self.test_size else self.X_unscaled
            else:
                X = self.X_test if self.test_size else self.X
            labels = ['y_pred', 'y']
            labels.extend(self.xlabel)
            ypred_df = pd.DataFrame(np.c_[self.y_pred, y, X.values], columns=labels)
            logger.info(ypred_df)

        if self.save:
            logger.info('save')
            self.summary_stats()

            if os.path.exists(self.xls_file):
                with pd.ExcelWriter(self.xls_file, mode='a') as writer:
                    ypred_df.to_excel(writer, sheet_name='y-pred')
            else:
                with pd.ExcelWriter(self.xls_file) as writer:
                    ypred_df.to_excel(writer, sheet_name='y-pred')
            self.report_classification_cm()

    def plot_cluster_report(self, type_set: str, report_type: str):
        """
        Plot silhoutte graph

        Parameters
        ----------
            type_set (str): set type either TRAINING_SET or TEST_SET constants
            ax (plt.Axes): specify axes of the plot
            report_type (str): set the report type as INTERCLUSTER_DISTANCE

        Reference
        _________
            https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
            https://www.scikit-yb.org/en/latest/api/cluster/icdm.html
        """
        report_type = self.modifier + '_' + report_type
        if type_set:
            report_type += '_' + type_set

        logger.info(f'report_type: {report_type}')

        X = y = None
        if type_set:

            logger.info(f'X_train: {self.X_train} X_test: {self.X_test}')
            logger.info(f'unscaled: X_train: {self.X_train_unscaled} X_test: {self.X_test_unscaled} ')

            if self.sc_X is not None:
                X_train = self.X_train_unscaled
                X_test = self.X_test_unscaled

            X = X_train if type_set == c.TRAINING_SET else X_test

            if self.sc_X is not None:
                y_train = self.y_train_unscaled
                y_test = self.y_test_unscaled

            y = y_train if type_set == c.TRAINING_SET else y_test

        else:

            X = self.X
            if self.sc_X:
                X = self.X_unscaled


            y = self.y
            if self.sc_y:
                y = self.y_unscaled

        estimator = eval(str(self.estimator))
        if report_type == c.INTERCLUSTER_DISTANCE:
            if self.has_y:
                intercluster_distance(estimator=estimator,X=X, y=y, colors='yellowbrick')
            else:
                intercluster_distance(estimator=estimator, X=X, colors='yellowbrick')
        elif report_type == c.SILHOUETTE:
            if self.has_y:
                silhouette_visualizer(estimator=estimator,X=X, y=y, colors='yellowbrick')
            else:
                silhouette_visualizer(estimator=estimator, X=X, colors='yellowbrick')

        plt.show()

        # file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(report_type, c.DTTM_FMT, c.GRAPHIC_EXT)
        # plt.savefig(file)


    def plot_elbow(self, k: int):
        """
        Plot silhoutte graph

        Parameters
        ----------
            k (int): number of clusters
        """
        fig, ax = plt.subplots(1, 1)
        visualizer = KElbowVisualizer(self.estimator, k=k)
        visualizer.fit(self.X)
        visualizer.show()
        self.n_cluster = visualizer.elbow_value_

        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_elbow', c.DTTM_FMT, c.GRAPHIC_EXT)
            fig.savefig(file)

    def plot_silhouette(self, type_set: str):
        """
        Plot silhoutte graph

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants
           ax (plt.Axes): specify axes of the plot

        Reference
        _________
            https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
        """
        fig, ax = plt.subplots(1, 1)
        report_type = self.modifier + '_silhouette'
        if type_set:
            report_type += '_' + type_set
        X = y = None
        if type_set:

            if self.sc_X is not None:
                X_train = self.X_train_unscaled
                X_test = self.X_test_unscaled

            X = X_train if type_set == c.TRAINING_SET else X_test

            if self.sc_X is not None:
                y_train = self.y_train_unscaled
                y_test = self.y_test_unscaled

            y = y_train if type_set == c.TRAINING_SET else y_test

        else:

            X = self.X
            if self.sc_X:
                X = self.X_unscaled

            y = self.y
            if self.sc_y:
                y = self.y_unscaled
            #y = self.y

        estimator = eval(str(self.estimator))

        if self.has_y:
            ax = silhouette_visualizer(estimator=estimator,X=X, y=y, colors='yellowbrick')
        else:
            ax = silhouette_visualizer(estimator=estimator, X=X, colors='yellowbrick')

        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(report_type, c.DTTM_FMT, c.GRAPHIC_EXT)
            fig.savefig(file)

        ax.show()

    @h.measure_performance
    def plot_cluster_scatter(self, type_set: str, ax: plt.Axes):
        """
        Function to plot the scatter graph for cluster

        Parameters
        ----------
           type_set (str): set type either TRAINING_SET or TEST_SET constants
           ax (plt.Axes): specify axes of the plot

        """
        logger.info(f'plot_scatter: {type_set}')
        logger.info(f'test_size: {self.test_size} {type(self.X)} {type(self.y)} ')
        # _, i = self.X.shape if self.test_size is None else self.X_train.shape
        # if i != 2:
        #     return

        X = y = X_train = y_train = X_test = y_test = None

        if type_set:

            X_train = self.X_train
            X_test = self.X_test

            if self.sc_X:
                X_train = self.X_train_unscaled
                X_test = self.X_test_unscaled

            y_train = self.y_train
            y_test = self.y_test

            if self.sc_y:
                y_train = self.y_train_unscaled
                y_test = self.y_test_unscaled

            print(f'y_train: {y_train} y_test: {y_test}')
            X = X_train if type_set == c.TRAINING_SET else X_test
            y = y_train if type_set == c.TRAINING_SET else y_test

        else:
            X = self.X
            if self.sc_X:
                X = self.X_unscaled
            y = self.y
            if self.sc_y:
                y = self.y_unscaled

        y = y if self.has_y else self.y_pred

        if self.modifier == 'KMeans' and not self.sc_X:
            center = self.estimator.cluster_centers_
            ax.scatter(center[:,0], center[:,1], marker='*', s=300, c='yellow', label='Centroids')
        sns.scatterplot(ax=ax, x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, edgecolor="black")

        self.ylabel = X.iloc[:, 1].name
        self.set_label(type_set, ax)


    @h.measure_performance
    def plot_cluster(self):
        """
        Function to plot the 2D cluster model using scatter plot
        """
        h.debug(f'plot_lr: {self.modifier}, {self.test_size}')

        if self.test_size is not None:

            self.fig, self.axs = plt.subplots(1, 2)
            self.plot_cluster_scatter(c.TRAINING_SET, self.axs[0])
            self.plot_cluster_scatter(c.TEST_SET, self.axs[1])
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in self.axs.flat:
                ax.label_outer()
        else:
            print('here1')
            self.fig, self.ax = plt.subplots(1, 1)
            self.plot_cluster_scatter(None, self.ax)

        plt.suptitle(f'Scatter Plot')
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.GRAPHIC_EXT)
            self.fig.savefig(file)

    @h.measure_performance
    def plot_csm(self):
        """
        Function to plot the 2D cluster model using scatter plot
        """

        if self.test_size is not None:

            self.fig, self.axs = plt.subplots(1, 2)
            self.plot_silhouette(c.TRAINING_SET, self.axs[0])
            self.plot_silhouette(c.TEST_SET, self.axs[1])
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in self.axs.flat:
                ax.label_outer()
        else:
            self.fig, self.ax = plt.subplots(1, 1)
            self.plot_silhouette(None, self.ax)
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_silhouette', c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)


    @h.measure_performance
    def plot_dendograph(self):
        """
        Function to plot the dendograph for hierarchical clustering
        """
        fig, ax = plt.subplots(1, 1)
        X = self.X
        if self.sc_X:
            X = self.X_unscaled

        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
        xlabel = self.file.split('.')[0]
        self.set_label(None, ax, ': Dendograph', xlabel, 'Euclidean distances')
        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier + '_dendo', c.DTTM_FMT, c.GRAPHIC_EXT)
            fig.savefig(file)

        plt.show()

    @h.measure_performance
    def plot(self):
        """
        Function to plot the cluster model using various plot
        """

        self.plot_pairplot('None')
        if self.test_size:
            self.plot_pairplot(c.TRAINING_SET)
            self.plot_pairplot(c.TEST_SET)

        # skip for Agglomerative due to data interjection of proxy with message "'AgglomerativeClustering' object has no attribute 'predict'"
        # but will work for other desktop or laptop, my laptop is always going thru proxy with switch
        if self.modifier == 'AgglomerativeClustering':
            self.plot_dendograph()
        elif not self.no_n_cluster:
            self.plot_silhouette(None)
            if self.test_size:
                self.plot_silhouette(c.TRAINING_SET)
                self.plot_silhouette(c.TEST_SET)

        # self.plot_cluster_report(None, c.SILHOUETTE)
        # if self.test_size:
        #     self.plot_cluster_report(c.TRAINING_SET, c.SILHOUETTE)
        #     self.plot_cluster_report(c.TRAINING_SET, c.SILHOUETTE)
        #
        # self.plot_cluster_report(None, c.INTERCLUSTER_DISTANCE)
        # if self.test_size:
        #     self.plot_cluster_report(c.TRAINING_SET, c.INTERCLUSTER_DISTANCE)
        #     self.plot_cluster_report(c.TRAINING_SET, c.INTERCLUSTER_DISTANCE)

        self.plot_cluster()

    def execute(self):
        """
        Function to execute the clustering providing plot
        """
        try:
            self.train_predict()
            self.plot()
        except (Exception) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            logger.error(message)