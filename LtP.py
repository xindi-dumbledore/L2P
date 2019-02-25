"""
Learning to Place Algorithm.

Learning to Place is an algirothm developed by Xindi Wang, Onur Varol and Tina Eliassi-Rad to tackle the problem of predicting heavy-tail distributed attributes.

The prediction process have two stages:
1) predict pairwise relationship between the test instance to all the train instance;
2) assign the placing and obtain the prediction via voting.

The code is written in the sklearn style, so it will be easy to incorporate with.

# Author: Xindi Wang <xindi.w1993@gmail.com>
"""


import numpy as np
from model_help_functions import *


class LearningToPlace():
    """The Learning to Place Class."""

    def __init__(self, method="voting", efficient="True", clf=None):
        """
        Initialization.

        args:
        Method - method using in stage 2. # we have a lot of methods, for simplicity, I'm only showing the voting
        efficient - whether train the model using efficient algorithm we designed
        clf - the classifier to train on pairwise relationship classification
        """
        self.method = method
        self.efficient = efficient
        self.clf = clf

    def create_comparison(self, X, y):
        """
        Create comparison between each pair in the data.

        Keyword arguments:
        X -- the feature matrix of the data
        y -- the target variable of the data
        """
        self.sort_index = np.argsort(y)[::-1]
        self.y = y
        self.X_sorted = X[self.sort_index]
        self.y_sorted = y[self.sort_index]
        if self.efficient:
            self.Xp, self.yp, self.pairs = create_pairwise_train_two_pair_efficient(
                self.X_sorted, self.y_sorted, self.sort_index, 100, 50)
        else:
            self.Xp, self.yp, self.pairs = create_pairwise_train_two_pair(
                self.X_sorted, self.y_sorted, self.sort_index)
        if self.method == "voting":  # obtain the intervals which is utlized in voting
            self.interval = []
            self.intervalEdges = sorted(list(set(list(self.y))), reverse=True)
            for i in range(len(self.intervalEdges) - 1):
                self.interval.append(
                    (self.intervalEdges[i], self.intervalEdges[i + 1]))

    def fit(self, X, y):
        """ Generate compare comparison and fit the classifier.

        Keyword arguments:
        X -- feature matrix
        y -- target variable
        """
        self.create_comparison(X, y)
        self.clf.fit(self.Xp, self.yp)

    def voting(self, ypPredict):
        """
        The voting method in Stage 2.

        Keyword arguments:
        ypPredict -- the predicted pairwise relationship between test instance and every train instance.
        Output:
        predict -- the prediction for the test instance
        """
        voteDist = dict(zip(self.interval, [0] * len(self.interval)))
        for i, compareResult in enumerate(ypPredict):
            # get the interval of the training instance
            intervalIndex = np.digitize(self.y_sorted[i], self.intervalEdges)
            # get the intervals on the left and on the right
            left = self.interval[:intervalIndex]
            right = self.interval[intervalIndex:]
            # choose the upvote and downvote area
            voted = left if compareResult < 0 else right
            downvoted = right if compareResult < 0 else left
            # do the vote
            for each_voted in voted:
                voteDist[each_voted] += 1
            for each_voted in downvoted:
                voteDist[each_voted] -= 1
        # obtain the interval with the most votes
        intervalMostVote = [(k, voteDist[k])
                            for k in sorted(voteDist, key=voteDist.get)][-1][0]
        # obtain the prediction
        predict = np.mean(list(intervalMostVote))
        return predict

    def create_pairwise_Xtest(self, X_test):
        """
        Create the pairwise comparison matrix for the test instance against all train instances.

        Keyword arguments:
        X_test -- test instance feature vector
        Output:
        Xp -- comparison matrix of features
        pairs -- index pair of each pair
        """
        Xp = []
        pairs = []
        for i in range(len(self.X_sorted)):
            Xp.append(np.hstack([self.X_sorted[i], X_test]))
            pairs.append((self.sort_index[i], -1))  # test_index = -1
        return Xp, pairs

    def predict(self, X_test):
        """
        Make the prediction for the test instance.

        Keyword arguments:
        X_test -- test instance feature vector
        Output:
        predict -- the prediction of the test instance
        """
        Xp_test, pairs_test = self.create_pairwise_Xtest(X_test)
        if self.method == "voting":
            # get pairwise relationship prediction
            ypPredict = self.clf.predict(Xp_test)
            # get the final prediction with voting
            predict = self.voting(ypPredict)
        else:
            raise NameError('Method %s not recognized' % self.method)
        return predict
