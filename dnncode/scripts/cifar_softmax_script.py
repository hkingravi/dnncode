import time
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

from dnncode.datasets import get_CIFAR10_data
from dnncode.classifiers import loss_softmax, Softmax, grad_check_sparse

data_dir = "../datasets/cifar-10-batches-py/"

# set seed
np.random.seed(0)
visualize = False

# matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

# convert to column major: will change to accommodate visualization later
X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
X_dev = X_dev.T
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
print 'dev data shape: ', X_dev.shape
print 'dev labels shape: ', y_dev.shape

# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(10, 3073) * 0.0001
loss, grad = loss_softmax(data=X_dev, labels=y_dev, weights=W, reg_val=0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print 'loss: ' + str(loss)
print 'sanity check: %f' % (-np.log(0.1))

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
f = lambda w: loss_softmax(data=X_dev, labels=y_dev, weights=W, reg_val=0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)


# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7]
regularization_strengths = [1e4]

params = list(itertools.product(learning_rates, regularization_strengths))
for curr_param in params:
    print "Computing training and validation results on (rate, reg): (" + str(curr_param[0]) + ", " \
          + str(curr_param[1]) + ")..."
    curr_softmax = Softmax()
    tic = time.time()
    loss_hist = curr_softmax.train(X_train, y_train, learning_rate=curr_param[0], reg=curr_param[1],
                                   num_iters=5000, verbose=True)
    toc = time.time()
    print 'Training time: %fs' % (toc - tic)
    y_train_pred = curr_softmax.predict(X_train)
    curr_train = np.mean(y_train == y_train_pred)
    print 'training accuracy: %f' % (curr_train,)
    y_val_pred = curr_softmax.predict(X_val)
    curr_val = np.mean(y_val == y_val_pred)
    print 'validation accuracy: %f' % (curr_val,)
    results[curr_param] = (curr_train, curr_val)
    if curr_val > best_val:
        best_val = curr_val
        best_softmax = curr_softmax

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy)

print 'best validation accuracy achieved during cross-validation: %f' % best_val

# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

# Visualize the learned weights for each class
w = best_softmax.W
w = w[:, :-1].T  # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()
