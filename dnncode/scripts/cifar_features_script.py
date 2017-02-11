import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from dnncode.datasets import load_CIFAR10
from dnncode.features import *
from dnncode.classifiers import LinearSVM, TwoLayerNet
from dnncode.utils import visualize_grid

data_dir = "../datasets/cifar-10-batches-py/"

# set seed
np.random.seed(0)
visualize = False

# matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = data_dir
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

num_color_bins = 30  # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

learning_rates = [5e-9]
regularization_strengths = [1e6]

results = {}
best_val = -1
best_svm = None

print X_val_feats.shape
print X_train_feats.shape

params = list(itertools.product(learning_rates, regularization_strengths))
for curr_param in params:
    print "Computing training and validation results on (rate, reg): (" + str(curr_param[0]) + ", " \
          + str(curr_param[1]) + ")..."
    curr_svm = LinearSVM()
    tic = time.time()
    loss_hist = curr_svm.train(X_train_feats.T, y_train, learning_rate=curr_param[0], reg=curr_param[1],
                               num_iters=5000, verbose=True)
    toc = time.time()
    print 'Training time: %fs' % (toc - tic)
    y_train_pred = curr_svm.predict(X_train_feats.T)
    curr_train = np.mean(y_train == y_train_pred)
    print 'training accuracy: %f' % (curr_train,)
    y_val_pred = curr_svm.predict(X_val_feats.T)
    curr_val = np.mean(y_val == y_val_pred)
    print 'validation accuracy: %f' % (curr_val,)
    results[curr_param] = (curr_train, curr_val)
    if curr_val > best_val:
        best_val = curr_val
        best_svm = curr_svm

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy)

print 'best validation accuracy achieved during cross-validation: %f' % best_val

# Evaluate your trained SVM on the test set
y_test_pred = best_svm.predict(X_test_feats.T)
test_accuracy = np.mean(y_test == y_test_pred)
print test_accuracy

# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()

print X_train_feats.shape

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

# tune hyperparameters
results = {}
best_val = -1
best_net = None  # store the best model into this
learning_rates = [1e-6]
regularization_strengths = [1e5, 5e5, 1e6]

params = list(itertools.product(learning_rates, regularization_strengths))
for curr_param in params:
    print "Computing training and validation results on (rate, hidden_size): (" + str(curr_param[0]) + ", " \
          + str(curr_param[1]) + ")..."
    curr_net = TwoLayerNet(input_dim, hidden_dim, num_classes)
    tic = time.time()
    loss_hist = curr_net.train(X_train_feats, y_train, X_val_feats, y_val, num_iters=3000, batch_size=200,
                               learning_rate=curr_param[0], learning_rate_decay=0.95, reg=curr_param[1], verbose=True)
    toc = time.time()
    print 'Training time: %fs' % (toc - tic)
    y_train_pred = curr_net.predict(X_train_feats)
    curr_train = np.mean(y_train == y_train_pred)
    print 'training accuracy: %f' % (curr_train,)
    y_val_pred = curr_net.predict(X_val_feats)
    curr_val = np.mean(y_val == y_val_pred)
    print 'validation accuracy: %f' % (curr_val,)
    results[curr_param] = (curr_train, curr_val)
    if curr_val > best_val:
        best_val = curr_val
        best_net = curr_net

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy)

print 'best validation accuracy achieved during cross-validation: %f' % best_val

# visualize the weights of the best network
show_net_weights(best_net)
test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print 'Test accuracy: ', test_acc


