# Machine Learning
: k-Nearest Neighbor/ Linear Regression/ Perceptron/ Logistic Regression/ Neural Networks/ k-means Clustering/ Gaussian Mixture Models/ Hidden Markov Models/ Principal Component Analysis

## Enviornments
numpy==1.13.1<br />
scipy==0.19.1<br />
matplotlib==2.0.2<br />
scikit-learn==0.19.0<br />
jupyter==1.0.0<br />

## Dataset
The dataset is stored in a JSON-formated file (images of handwritten digits from 0 to 9).
Its training, validation, and test splits using the keys ‘train’, ‘valid’, and ‘test’, respectively.
This set is a list with two elements: x['train'][0] containing the features of size N(samples) × D(dimension of features),
and x['train'][1] containing the corresponding labels of size N.

## Code
1. __knn.py__: k-Nearest Neighbor 

2. __lr.py__: Linear Regression 

3. __preceptron.py__: perceptron 

4. __logistic.py__: Logistic Regression
   - __binary classification__: 'binary_train' and 'binary_predict' function will generate 'logistic_binary.out' file by 'logistic_binary.sh' script
   - __multiclassification__: 'OVR_train' and 'OVR_preduct' function will perform one-versus-rest classification and generate 'logistic_multiclass.out' file by 'logistic_multiclass.sh' script

5. __pegasos.py__: Stochastic gradient based solver for linear SVM
   - A linear SVM classifiere using the Pegasos algorithm

6. __boosting.py__: Boosting constructs a strong binary classifier based on iteratively adding one weak (binary) classifier into it. A weak classifier is learned to maximize the weighted training accuracy at the corresponding iteration, and the weight of each training example is updated after every iteration.
   - __AdaBoost__: boosting method using the exponential loss function
   - __LogitBoost__: boosting method using the logloss

7. __decision_tree.py__: Decision Tree classifier

8. __kmeans__: k-Means Clustering
   - __'toy_dataset_kmeans.png'__ was found the means of 4 cluseters by kmeas trained program from the original image which is __'toy_dataset_predicted.png'__
   - Image compression by k-means: __'baboon_original.png'__ -> __'baboon_compressed.png'__

9. __gmm__: Gaussian Mixture Models 
   - Parameters of GMM can be estimated using EM algorithm
   - __'toydataset_gmm.png'__: Gaussian Mixture model on toy dataset
   - GMM is that we can sample from the learnt distribution and generate new examples which should be similar to the actual data (__'samples_k_means.png'__, __'samples_random.png'__)

10. __hmm.py__: Hidden Markov Models

11. __pca.py__: Principal Component Analysis
