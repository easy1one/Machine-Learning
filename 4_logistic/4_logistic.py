from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    # Set w_ and x_
    ones = np.matrix(np.ones(N));
    ones = np.reshape(ones,(N,1))
    X_ = np.append(ones,X,axis=1)
    w_ = np.insert(w,0,b)

    for t in range(max_iterations):
        # Initi
        gradi = np.zeros(D+1)

        # Update Gradient
        mySigmo = np.vectorize(sigmoid)
        wTx = mySigmo(np.dot(w_, X_.T))
        gradi = np.dot(np.subtract(wTx, y), X_)

        # Update weight
        w_ = w_ - (step_size * (np.array(gradi) / N))

    b = w_[0][0]
    w_ = np.delete(w_,0)
    w=w_

    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N) 

    for n in range(N):
        preds[n] = 1 if(sigmoid(b + np.inner(w,X[n])) >= 0.5) else 0

    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    # Set X_
    ones = np.matrix(np.ones(N));
    ones = np.reshape(ones, (N, 1))
    X_ = np.append(ones, X, axis=1)

    # Set w_
    b = np.matrix(b)
    b = np.reshape(b, (C, 1))
    w_ = np.append(b, w, axis=1)

    # Set Y_
    Y_ = np.zeros((N, C))
    for n in range(N):
        Y_[n][y[n]] = 1

    for t in range(max_iterations):
        # Initi
        gradi = np.zeros(D + 1)

        # Update Gradient
        e_x = np.exp(np.dot(w_, X_.T))
        wTx = e_x / e_x.sum(axis=0)
        gradi = np.dot(np.subtract(wTx, Y_.T), X_)
        w_ = w_ - (step_size * (gradi / N))

    b = w_[:, 0]
    b = np.transpose(b).tolist()[0]
    b = np.array(b)
    w_ = np.delete(w_, 0, 1)
    w = w_

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    # Set X_
    ones = np.matrix(np.ones(N));
    ones = np.reshape(ones, (N, 1))
    X_ = np.append(ones, X, axis=1)

    # Set w_
    b = np.matrix(b)
    b = np.reshape(b, (C, 1))
    w_ = np.append(b, w, axis=1)

    # Get argmax for class
    wTx = np.dot(w_, X_.T)
    for col in range(N):
        preds[col] = np.argmax(wTx[:,col])

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    """
    TODO: add your code here
    """

    # Set X_
    ones = np.matrix(np.ones(N));
    ones = np.reshape(ones, (N, 1))
    X_ = np.append(ones, X, axis=1)

    # Set w_
    b = np.matrix(b)
    b = np.reshape(b, (C,1))
    w_ = np.append(b, w, axis=1)

    # Set Y_
    Y_ = np.zeros((N, C))
    for n in range(N):
        Y_[n][y[n]] = 1

    for t in range(max_iterations):
        # Initi
        gradi = np.zeros(D+1)

        # Update Gradient
        mySigmo = np.vectorize(sigmoid)
        wTx = mySigmo(np.dot(w_, X_.T))
        gradi = np.dot(np.subtract(wTx, Y_.T), X_)
        w_ = w_ - (step_size * (np.array(gradi) / N))

    b = w_[:,0]
    b = np.transpose(b).tolist()[0]
    b = np.array(b)
    w_ = np.delete(w_,0,1)
    w = w_

    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)
    
    for n in range(N):
        classes = np.zeros(C)
        for k in range(C):
            classes[k] = sigmoid(b[k] + np.inner(w[k],X[n]))
        preds[n] = np.argmax(classes)

    assert preds.shape == (N,)
    return preds


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        