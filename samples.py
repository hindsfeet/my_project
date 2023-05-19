""""
This is for sampling implementation

MCCT (initial)/ Minnie Cherry Chua Tan 29-Oct-21 Added handling for UCB, Thompson Sampling
MCCT (initial)/ Minnie Cherry Chua Tan 27-May-22 Updated to put the rewards and clicks in stats.xlsx, added ads_selected_count()
"""

# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

from model import Model
from constant import Constant as c
from helper import Helper as h

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import math
import random

logger = logging.getLogger(__name__)

class Samples(Model):
    @h.measure_performance
    def train_predict(self):
        """
        Function to train and predict the model using the datasets
        """
        h.debug('train_predict')
        if self.modifier == c.UCB:
            self.upper_confidence()
        elif self.modifier == c.THOMPSONSAMPLING:
            self.thompson_sampling()



    @h.measure_performance
    def upper_confidence(self):
        """
        Function to get the upper confidence bounds using rewards based on the selection

        References
        ----------
        Udemy Machine Learning A-Z Hands-On Python & R In Data Science Course by Kirill Eremenko
        """
        N, d = self.dataset.shape
        self.result = pd.DataFrame(0,index=range(N), columns=range(d+1))
        self.ads_selected = []
        numbers_of_selections = [0] * d
        sums_of_rewards = [0] * d
        total_reward = 0
        for n in range(0, N):
            ad = 0
            max_upper_bound = 0
            for i in range(0, d):
                if (numbers_of_selections[i] > 0):
                    average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                    delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    ad = i
            self.ads_selected.append(ad)
            numbers_of_selections[ad] += 1
            reward = self.dataset.values[n, ad]
            sums_of_rewards[ad] += reward
            total_reward += reward

            self.ads_selected_count(n, total_reward)

        self.result.rename(columns={self.result.columns[-1]: "Rewards"}, inplace=True)
        print(self.result)
        if self.save:
            print(self.stat_file)
            with pd.ExcelWriter(self.xls_file) as writer:
                self.result.to_excel(writer, sheet_name='stats')


    @h.measure_performance
    def thompson_sampling(self):
        """
        Function to get the thompson sampling using rewards based on the selection

        References
        ----------
        Udemy Machine Learning A-Z Hands-On Python & R In Data Science Course by Kirill Eremenko
        """
        N, d = self.dataset.shape

        self.result = pd.DataFrame(0,index=range(N), columns=range(d+1))
        self.ads_selected = []
        numbers_of_rewards_1 = [0] * d
        numbers_of_rewards_0 = [0] * d
        total_reward = 0
        for n in range(0, N):
            ad = 0
            max_random = 0
            for i in range(0, d):
                random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
                if random_beta > max_random:
                    max_random = random_beta
                    ad = i
            self.ads_selected.append(ad)
            reward = self.dataset.values[n, ad]
            if reward == 1:
                numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
            else:
                numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
            total_reward = total_reward + reward

            self.ads_selected_count(n, total_reward)

        self.result.rename(columns={self.result.columns[-1]: "Rewards"}, inplace=True)
        print(self.result)
        if self.save:
            print(self.stat_file)
            with pd.ExcelWriter(self.xls_file) as writer:
                self.result.to_excel(writer, sheet_name='stats')

    @h.measure_performance
    def ads_selected_count(self, row: int, total_reward: int):
        """
        Function to ads selection count
        """
        uniq = np.unique(self.ads_selected)
        self.result.iloc[row,-1] = total_reward
        # self.ads_selected.apply(lambda col: col.nunique()).apply(lambda x: np.count(x), result_type='broadcast')
        for col in uniq:
            self.result.iloc[row,col] = self.ads_selected.count(col)

    @h.measure_performance
    def plot(self):
        """
        Function to plot the model
        """
        # if self.modifier == c.THOMPSONSAMPLING:
        #     print(f'{self.ads_selected}')
        #     exit(0)

        ax = sns.displot(self.ads_selected, kde=True)
        xlabel = self.graph_dict[c.XLABEL] if c.XLABEL in self.graph_dict.keys() else self.X.columns[0]
        ylabel = self.graph_dict[c.YLABEL] if c.YLABEL in self.graph_dict.keys() else 'Count'
        ax.set_axis_labels(xlabel, ylabel)
        plt.title(f"{self.modifier}: {self.filename.title()} Histogram")
        plt.tight_layout()

        if self.graph:
            file = h.get_path(self.path) + c.DIR_DELIM + h.get_file(self.modifier, c.DTTM_FMT, c.GRAPHIC_EXT)
            print(f'file: {file}')
            plt.savefig(file)
        plt.show()

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
