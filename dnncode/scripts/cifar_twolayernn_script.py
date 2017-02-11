import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from dnncode.datasets import load_CIFAR10
from dnncode.classifiers import TwoLayerNet, eval_numerical_gradient_orig
from dnncode.utils import visualize_grid

data_dir = "../datasets/cifar-10-batches-py/"

# set seed
np.random.seed(0)
visualize = False

# matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """
    Returns relative error.

    :param x:
    :param y:
    :return:
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = "../datasets/cifar-10-batches-py/"
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

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

net = init_toy_model()
X, y = init_toy_data()

print X.shape, y.shape

# we switch to row major convention

# implement forward pass
scores = net.loss(X)
print 'Your scores:'
print scores
print
print 'correct scores:'
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print correct_scores
print

# The difference should be very small. We get < 1e-7
print 'Difference between your scores and correct scores:'
print np.sum(np.abs(scores - correct_scores))

loss, _ = net.loss(X, y, reg=0.1)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print 'Difference between your loss and correct loss:'
print np.sum(np.abs(loss - correct_loss))

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.1)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.1)[0]
    print "Querying f with " + param_name
    print f(net.params[param_name])
    net.params[param_name] += 0.2
    print f(net.params[param_name])
    net.params[param_name] -= 0.2
    print "\n"
    param_grad_num = eval_numerical_gradient_orig(f, net.params[param_name], verbose=True)
    print "Gradients for %s parameter: " % param_name
    print "Numerical: " + str(param_grad_num)
    print "Analytical: " + str(grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

net = init_toy_model()
stats = net.train(X, y, X, y, learning_rate=1e-1, reg=1e-5,
                  num_iters=100, verbose=False)

print 'Final training loss: ', stats['loss_history'][-1]

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print 'Validation accuracy: ', val_acc

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

show_net_weights(net)

# tune hyperparameters
results = {}
best_val = -1
best_net = None  # store the best model into this
learning_rates = [1e-3, 5e-4]
hidden_sizes = [100, 200, 500]

params = list(itertools.product(learning_rates, hidden_sizes))
for curr_param in params:
    print "Computing training and validation results on (rate, hidden_size): (" + str(curr_param[0]) + ", " \
          + str(curr_param[1]) + ")..."
    curr_net = TwoLayerNet(input_size, curr_param[1], num_classes)
    tic = time.time()
    loss_hist = curr_net.train(X_train, y_train, X_val, y_val, num_iters=3000, batch_size=200,
                               learning_rate=curr_param[0], learning_rate_decay=0.95, reg=0.7, verbose=True)
    toc = time.time()
    print 'Training time: %fs' % (toc - tic)
    y_train_pred = curr_net.predict(X_train)
    curr_train = np.mean(y_train == y_train_pred)
    print 'training accuracy: %f' % (curr_train,)
    y_val_pred = curr_net.predict(X_val)
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
test_acc = (best_net.predict(X_test) == y_test).mean()
print 'Test accuracy: ', test_acc

