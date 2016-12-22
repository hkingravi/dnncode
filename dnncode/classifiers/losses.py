"""
Loss functions.
"""
import numpy as np


def loss_hinge_unvec(x, y, W):
    """
    Unvectorized version of SVM hinge-loss.

    :param x: column vector representing image, e.g. 3073 x 1 in CIFAR-10.
    :param y: integer giving index of correct class, e.g. between 0 and 9 in CIFAR-10.
    :param W: weight matrix, e.g. 10 x 3073 in CIFAR-10.
    :return:
    """
    delta = 1.0  # see notes about delta later in this section
    scores = W.dot(x)  # scores becomes of size 10 x 1, the scores for each class
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes, e.g. 10
    loss_i = 0.0
    for j in xrange(D):  # iterate over all wrong classes
        if j == y:
            # skip for the true class to only loop over incorrect classes
            continue
        # accumulate loss for the i-th example
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def loss_hinge_partvec(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    scores = W.dot(x)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def loss_hinge(data, labels, weights):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    delta = 1.0
    scores = weights.dot(data.T)  # compute dot product
    correct_scores = scores[labels, np.arange(labels.shape[0])]
    margins = np.maximum(0, scores - correct_scores + delta)
    margins[labels, np.arange(labels.shape[0])] = 0
    loss_i = np.sum(np.sum(margins, axis=0))
    return loss_i


def loss_hinge_reg(data, labels, weights, reg_val=0):
    """

    :param data: holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    :param labels: array of integers specifying correct class (e.g. 50,000 array)
    :param weights: are weights (e.g. 10 x 3073)
    :param reg_val: regularization value
    :return:
    """
    """
    Regularized fully-vectorized implementation of  hinge-loss:
    - X
    - y is
    - W
    """
    delta = 1.0
    nsamp = labels.shape[0]
    scores = weights.dot(data)  # compute dot product
    correct_scores = scores[labels, np.arange(labels.shape[0])]
    margins = np.maximum(0, scores - correct_scores + delta)
    margins[labels, np.arange(labels.shape[0])] = 0
    loss_i = (1/float(nsamp))*np.sum(np.sum(margins, axis=0)) + reg_val*np.sum(np.sum(weights ** 2, axis=0))
    return loss_i


def loss_softmax_partvec(x, y, W):
    """
    Implement softmax loss on multiclass data using normalization trick to avoid
    over/underflow.
    """
    scores = W.dot(x)
    scores -= np.max(scores)
    p = np.exp(scores)/np.sum(np.exp(scores))
    return p[y]


def loss_softmax(data, labels, weights, reg_val=0):
    """
    Implement softmax loss on multiclass data using normalization trick to avoid
    over/underflow. Here, N denotes number of samples, D the dimensionality of
    the data (including bias), and K the number of classes.

    :param data: N x D data matrix
    :param labels: N dim vector
    :param weights: K x D weight matrix
    :param reg_val: regularization value
    :return:
    """
    nsamp = labels.shape[0]
    scores = weights.dot(data.T)
    scores -= np.max(scores, axis=0)
    p = np.exp(scores)/np.sum(np.exp(scores), axis=0)
    loss_val = (1/float(nsamp))*p[labels, np.arange(labels.shape[0])] + (reg_val / 2 * float(nsamp)) * np.sum(np.sum(weights, axis=0))
    return loss_val

