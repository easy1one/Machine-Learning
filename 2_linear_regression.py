from __future__ import division, print_function

from typing import List

import numpy
import scipy

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):

        ones = numpy.ones((len(features), 1))
        self.features_train = numpy.concatenate((ones, numpy.asmatrix(features)), axis = 1)
        self.values_train = values
        self.weights = self.get_weights()

    def predict(self, features: List[List[float]]) -> List[float]:

        ones = numpy.ones((len(features),1))
        features_pred = numpy.concatenate((ones, numpy.asmatrix(features)), axis = 1)
        return numpy.dot(self.weights,features_pred.T).tolist()[0]

    def get_weights(self) -> List[float]:
        return numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(self.features_train.T, self.features_train)), self.features_train.T), self.values_train)


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):

        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):

        ones = numpy.ones((len(features), 1))
        self.features_train = numpy.concatenate((ones, numpy.asmatrix(features)), axis = 1)
        self.values_train = values
        self.weights = self.get_weights()

    def predict(self, features: List[List[float]]) -> List[float]:

        ones = numpy.ones((len(features),1))
        features_pred = numpy.concatenate((ones, numpy.asmatrix(features)), axis = 1)
        return numpy.dot(self.weights,features_pred.T).tolist()[0]

    def get_weights(self) -> List[float]:

        length = len(self.features_train.tolist()[0])
        regularizer = self.alpha *(numpy.array(numpy.identity(length)))
        return numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.add(numpy.dot(self.features_train.T, self.features_train),regularizer)), self.features_train.T), self.values_train)

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
