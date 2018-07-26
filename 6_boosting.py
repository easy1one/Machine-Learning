import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
	
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:

		N = len(features)
		sum = np.zeros(N,)

		for i in range(self.T):
			h = self.clfs_picked[i].predict(features)
			sum  = sum + self.betas[i] * np.array(h)
		for i, item in enumerate(sum):
			sum[i] = 1 if item > 0 else -1
		return list(sum.flatten())
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):

		N = len(features)
		w = np.divide(np.ones(N,), N)

		for t in range(self.T):
			# 1. Find ht by argmin
			h_t = None
			min_sum = float('inf')
			for clf in self.clfs:
				sum = 0.0
				h_labels = clf.predict(features)
				for i, h_label in enumerate(h_labels):
					if labels[i] != h_label:
						sum += w[i]
				if sum < min_sum:
					h_t = clf
					min_sum = sum

			# 2. error and beta
			beta = np.log((1-min_sum)/min_sum) / 2.0

			# 3. compute w_t+1
			h_t_labels = h_t.predict(features)
			for n, h_t_label in enumerate(h_t_labels):
				if labels[n] != h_t_label:
					w[n] *= np.exp(beta) 
				else:
					w[n] *= np.exp(-beta)
			w = np.divide(w, np.sum(w))
			#print(w)
			self.betas.append(beta)
			self.clfs_picked.append(h_t)

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return


	def train(self, features: List[List[float]], labels: List[int]):

		N = len(features)
		pi = np.divide(np.ones(N), 2)
		f = np.zeros(N,)

		for t in range(self.T):
			# 1. w
			w = np.multiply(pi,(1-pi))

			# 2. z
			up = np.divide((np.array(labels)+1),2)-pi
			z = np.divide(up,w)

			# 3. ht by argmin
			h_t = None
			min_sum = float('inf')
			for clf in self.clfs:
				sum = 0.0
				h_labels = clf.predict(features)
				sum = np.dot(w,np.power(z-h_labels,2))
				if sum < min_sum:
					h_t = clf
					min_sum = sum

			h_t_labels = h_t.predict(features)
			f = f + np.array(h_t_labels)/2
			pi = np.divide(1,1+np.exp((-2)*f))

			self.betas.append(1.0/2)
			self.clfs_picked.append(h_t)


	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	