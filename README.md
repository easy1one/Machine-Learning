# Machine Learning
: k-Nearest Neighbor/ Linear Regression/ Perceptron/ Logistic Regression/ Neural Networks/ k-means Clustering/ Gaussian Mixture Models/ Hidden Markov Models/ Principal Component Analysis

## Enviornments
numpy==1.13.1
scipy==0.19.1
matplotlib==2.0.2
scikit-learn==0.19.0
jupyter==1.0.0

## Dataset
The dataset is stored in a JSON-formated file (images of handwritten digits from 0 to 9).
Its training, validation, and test splits using the keys ‘train’, ‘valid’, and ‘test’, respectively.
This set is a list with two elements: x['train'][0] containing the features of size N(samples) × D(dimension of features),
and x['train'][1] containing the corresponding labels of size N.

## Code
1. knn.py: k-Nearest Neighbor 

2. lr.py: Linear Regression 

3. preceptron.py: perceptron 

4. logistic.py: Logistic Regression
- binary classification: 'binary_train' and 'binary_predict' function will generate 'logistic_binary.out' file by 'logistic_binary.sh' script
- multiclassification: 'OVR_train' and 'OVR_preduct' function will perform one-versus-rest classification and generate 'logistic_multiclass.out' file by 'logistic_multiclass.sh' script

5. pegasos.py: Stochastic gradient based solver for linear SVM
- A linear SVM classifiere using the Pegasos algorithm

6. boosting.py: Boosting constructs a strong binary classifier based on iteratively adding one weak (binary) classifier into it. A weak classifier is learned to maximize the weighted training accuracy at the corresponding iteration, and the weight of each training example is updated after every iteration.
- AdaBoost: boosting method using the exponential loss function
- LogitBoost: boosting method using the logloss

7. decision_tree.py: Decision Tree classifier

8. kmeans: k-Means Clustering
- 'toy_dataset_kmeans.png' was found the means of 4 cluseters by kmeas trained program from the original image which is 'toy_dataset_predicted.png' 
- Image compression by k-means: 'baboon_original.png' -> 'baboon_compressed.png'

9. gmm: Gaussian Mixture Models 
- Parameters of GMM can be estimated using EM algorithm
- 'toydataset_gmm.png': Gaussian Mixture model on toy dataset
- GMM is we can sample from the learnt distribution and generate new examples which should be similar to the actual data ('samples_k_means.png', 'samples_random.png')

10. hmm.py: Hidden Markov Models

11. pca.py: Principal Component Analysis
