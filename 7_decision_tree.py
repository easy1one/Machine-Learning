import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples
					            b1 b2
					  branches:[[2,2],  A
					  			[4,0]]  B
			'''
			branch_sum = branches.sum(axis=0)
			total = np.sum(branch_sum)
			total_frac = np.divide(branch_sum,total)
			sum_res = np.zeros((len(branches), len(branches[0])))
			for i in range(len(branches)):
				for j in range(len(branches[0])):
					if(branch_sum[j]!=0):
						sum_res[i][j]=branches[i][j]/branch_sum[j]
			for i in range(len(sum_res)):
				for j in range(len(sum_res[0])):
					if sum_res[i][j]==0:
						sum_res[i][j]=0
					else:
						sum_res[i][j]=sum_res[i][j]*np.log(sum_res[i][j])

			sum_ = sum_res.sum(axis=0)
			res_ = np.multiply(sum_,total_frac)
			res = (-1)*np.sum(res_) if (np.sum(res_)!=0) else 0
			return res
		
		feature_entropy = np.zeros(len(self.features[0]))

		for idx_dim in range(len(self.features[0])):
			dim_feature = np.array(self.features)[:,[idx_dim]].flatten()
			branch_matrix = np.zeros((self.num_cls, int(np.max(self.features))+1))
			for idx, branch_num in enumerate(dim_feature):
				branch_matrix[self.labels[idx]][int(branch_num)] += 1
			feature_entropy[idx_dim] = conditional_entropy(branch_matrix)

		feature_entropy = feature_entropy.flatten()
		min_dim_idx = 0
		for i in range(len(feature_entropy)):
			if feature_entropy[min_dim_idx] > feature_entropy[i]:
				min_dim_idx = i

		self.dim_split = min_dim_idx
		self.feature_uniq_split = np.unique(np.array(self.features)[:,min_dim_idx]).tolist()

		# Split child
		def partition(arr):
			return {tmp: (arr==tmp).nonzero()[0] for tmp in np.unique(arr)}
		maps = partition(np.array(self.features)[:,min_dim_idx])

		for key, val in maps.items():
			labels_subset = np.array(self.labels).take(val, axis=0).tolist()
			features_subset = np.array(self.features).take(val, axis=0).tolist()
			
			child = TreeNode(features_subset, labels_subset, np.max(labels_subset)+1)

			# values in x labels are same => set it False
			first_row = features_subset[0]
			result = np.all(np.array(features_subset)==np.array(first_row), axis=1)
			if(np.all(result)):
				child.splittable=False

			self.children.append(child)

		# Add children
		for child in self.children:
			if child.splittable:
				child.split()


	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



