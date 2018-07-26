from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

	def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
		'''
			Args : 
			nb_features : Number of features
			max_iteration : maximum iterations even if it is not converged
		'''
		self.nb_features = 2
		self.w = [0 for i in range(0,nb_features+1)]
		self.margin = margin
		self.max_iteration = max_iteration

	def train(self, features: List[List[float]], labels: List[int]) -> bool:
		'''
			Args  : 
			features : List of features. First element of each feature vector is 1 which is bias
			labels : label of each feature [-1,1]
			
			Returns : 
				True/ False : return True if the algorithm converges else False. 
		'''
		e = 0.00000001
		for times in range(self.max_iteration):
			#self.reset()
			flag = True
			div = np.linalg.norm(self.w) + e
			rand_index = list(range(len(features)))
			np.random.shuffle(rand_index)
			for i in range(len(rand_index)):
				cur_margin = np.dot(self.w, features[rand_index[i]]) / div
				if (labels[rand_index[i]] * np.dot(self.w, features[rand_index[i]]) < 0 or (-self.margin/2 < cur_margin < self.margin/2)):
					# Mistake happend -> Update weights
					flag = False
					self.w = np.add(self.w, labels[rand_index[i]] * np.array(features[rand_index[i]]))
			if flag == True:
				return True
		return False

	
	def reset(self):
		self.w = [0 for i in range(0,self.nb_features+1)]
		
	def predict(self, features: List[List[float]]) -> List[int]:
		'''
			Args  : 
			features : List of features. First element of each feature vector is 1 
			to account for bias
			
			Returns : 
				labels : List of integers of [-1,1] 
		'''
		labels_pred = list()
		for i in range(len(features)):
			if np.dot(self.w, features[i]) < 0:
				labels_pred.append(-1)
			else:
				labels_pred.append(1)
		return labels_pred

	def get_weights(self) -> List[float]:
		return self.w
	
