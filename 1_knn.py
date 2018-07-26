from __future__ import division, print_function
from typing import List, Callable

import numpy
import scipy

class KNN:
    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.features_train = features
        self.labels_train = labels

    def predict(self, features: List[List[float]]) -> List[int]:
    	label_predict = list()

    	for index, item in enumerate(features):
    		distance_list = list()
    		vote = [0]*2
    		for index_train, item_train in enumerate(self.features_train):
    			cur_dist = self.distance_function(item, item_train)
    			distance_list.append((cur_dist,index_train))

    		distance_list.sort(key=lambda x:x[0])

    		for i in range(self.k):
    			if self.labels_train[distance_list[i][1]] == 0:
    				vote[0] += 1
    			elif self.labels_train[distance_list[i][1]] == 1:
    				vote[1] += 1

    		if vote[0] > vote[1]:
    			label_predict.append(0)
    		else:
    			label_predict.append(1)

    	return label_predict


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
