"""
This is for rules model implementation

MCCT (initial)/ Minnie Cherry Chua Tan 22-Oct-21 Added handling for APRIORI in train_predict()
MCCT (initial)/ Minnie Cherry Chua Tan 23-Oct-21 Added handling for EClAT, APRIORI in plot()
MCCT (initial)/ Minnie Cherry Chua Tan 04-Mar-22 ax.bar_label becomes unavailable in the Python binary,
    added the xls for each rule
MCCT (initial)/ Minnie Cherry Chua Tan 05-Mar-22 Replace the unavailable bar_label codes
MCCT (initial)/ Minnie Cherry Chua Tan 08-Mar-22 Added metrics_rank() for the XLSX metrics, updated the hardcoding of n_largest
MCCT (initial)/ Minnie Cherry Chua Tan 09-Mar-22 Updated the formula used for n of n_largest to cater for even values
"""

# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui - my father,
# born in China, grew up in HK, after migrated in Philippines at 26-28 to marry my mother), my aut networkID is xrx5385 when I was student

__author__ = 'Minnie Tan'

from model import Model
from constant import Constant as c
from helper import Helper as h
from apyori import apriori

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import math
from numpy import median

logger = logging.getLogger(__name__)

class Rules(Model):
    @h.measure_performance
    def train_predict(self):
        """
        Function to train and predict the model using the datasets
        """
        print(f'train and predict:')
        dataset = self.dataset.astype(str)
        print(dataset)
        transactions = []
        i, j = dataset.shape
        transactions = np.ravel(dataset).reshape(i,j)
        h.debug(f'transactions: {transactions}')

        dataset = self.dataset.astype(str)
        transactions = []
        i, j = dataset.shape
        transactions = np.ravel(dataset).reshape(i, j)
        h.debug(f'transactions: {transactions}')

        min_support = self.model_dict[c.MIN_SUPPORT] if hasattr(self.model_dict,c.MIN_SUPPORT) else 0.003
        min_confidence = self.model_dict[c.MIN_CONFIDENCE] if hasattr(self.model_dict,c.MIN_CONFIDENCE) else 0.2
        min_lift = self.model_dict[c.MIN_LIFT] if hasattr(self.model_dict, c.MIN_LIFT) else 3
        min_length = self.model_dict[c.MIN_LENGTH] if hasattr(self.model_dict, c.MIN_LENGTH) else 2
        max_length = self.model_dict[c.MAX_LENGTH] if hasattr(self.model_dict, c.MAX_LENGTH) else 2

        rules = apriori(transactions=transactions, min_support=min_support, min_confidence=min_confidence,
                                      min_lift=min_lift, min_length=min_length, max_length=max_length)

        result = list(rules)
        print(result)

        df = pd.DataFrame(self.inspect(result),columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
        print(df)

        print(f'modifier: {self.modifier}')
        n = math.ceil(len(df) / 2)
        print(f'n:{n}')

        ## Displaying the results sorted by descending lifts
        if self.modifier == c.APRIORI:
            rdf = df.nlargest(n=n, columns='Lift')
            print(rdf)

            df1 = pd.DataFrame(
                data={c.ITEM: rdf.iloc[:, 0] + '+' + rdf.iloc[:, 1], c.METRICS: 'Support', c.VALUE: rdf.iloc[:, 2]})
            df1 = self.metrics_rank(df1, df1['value'])

            df2 = pd.DataFrame(
                data={c.ITEM: rdf.iloc[:, 0] + '+' + rdf.iloc[:, 1], c.METRICS: 'Confidence', c.VALUE: rdf.iloc[:, 3]})
            df2 = self.metrics_rank(df2, df2['value'])

            df3 = pd.DataFrame(
                data={c.ITEM: rdf.iloc[:, 0] + '+' + rdf.iloc[:, 1], c.METRICS: 'Lift', c.VALUE: rdf.iloc[:, 4]})
            df3 = self.metrics_rank(df3, df3['value'])
            self.result = df1.append(df2).append(df3)

        elif self.modifier == c.ECLAT:
            rdf = df.nlargest(n=n, columns='Support')
            self.result = pd.DataFrame(
                data={c.ITEM: rdf.iloc[:, 0] + '+' + rdf.iloc[:, 1], c.METRICS: c.SUPPORT, c.VALUE: rdf.iloc[:, 2]})
            self.result = self.metrics_rank(self.result, self.result['value'])

        print(self.result)
        if self.save:
            print(self.stat_file)
            with pd.ExcelWriter(self.xls_file) as writer:
                self.result.to_excel(writer, sheet_name='stats')

    @h.measure_performance
    def metrics_rank(self, df: pd.DataFrame, value: pd.Series) -> pd.DataFrame:
        """
        Function to add the value rank based on default rank, max, bottom and percentage

        Parameters
        ----------
           df (pd.Dataframe): dataframe of the value column
           value (pd.Series): value of the data column

        Return
        ------
           df (dataframe): dataframe with the added rank metrics

        References
        __________
           rank - https://pandas.pydata.org/docs/reference/api/pandas.Series.rank.html
        """
        logger.debug(f'original dataframe: {df}')
        df = df.sort_values(by=['value'], ascending=False)
        logger.debug(f'sorted dataframe: {df}')

        df['default_rank'] = value.rank()
        df['max_rank'] = value.rank(method='max')
        df['NA_bottom'] = value.rank(na_option='bottom')
        df['pct_rank'] = value.rank(pct=True)

        return df

    @h.measure_performance
    def inspect(self, results: list) -> list:
        """
        Function to parse the rule results into a list

        Parameters
        ----------
            results (list): raw result list

        Return
        ------
           (list): organized result list

        References
        ----------
           Udemy: Machine Learning A-Z Hands-On Python & R In Data Science - Rules
           https://www.udemy.com/course/machinelearning/learn/lecture/19799
        """
        lhs = []
        rhs = []
        supports = []
        confidences = []
        lifts = []
        for r in results:
            print("====")
            lhs.append(tuple(r[2][0][0])[0])
            rhs.append(tuple(r[2][0][1])[0])
            supports.append(r[1])
            confidences.append(r[2][0][2])
            lifts.append(r[2][0][3])
        return list(zip(lhs, rhs, supports, confidences, lifts))

    @h.measure_performance
    def plot(self):
        """
        Function to plot the model

        References
        __________
            values in barplot - https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
            https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html
            https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
            barplot syntax - https://seaborn.pydata.org/generated/seaborn.barplot.html
            rotate axis - https://drawingfromdata.com/seaborn/matplotlib/visualization/rotate-axis-labels-matplotlib-seaborn.html
        """
        print('plot')
        print(self.result)
        self.fig, ax = plt.subplots()
        ax = sns.barplot(x="item", y="value", hue="metrics", data=self.result, estimator=median)
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)
        title = self.modifier.title() + ' Barplot: ' + self.filename.title()
        ax.set_title(f'{title}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

        if self.graph and (hasattr(self, 'fig')):
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            self.fig.savefig(file)

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
