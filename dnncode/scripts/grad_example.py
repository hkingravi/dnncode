import time
import os
import numpy as np
import matplotlib.pyplot as plt
from dnncode.utils import unpickle, gen_image
from dnncode.classifiers import losses, gradients


def lin_class(data, weights):
    """
    Compute linear classifier on data.

    :param data:
    :param weights:
    :return:
    """
    return weights.dot(data)


def data_loss_hinge(weights_in):
    """
    Compute loss on data given an input of weights.

    :param weights_in:
    :return:
    """
    return losses.loss_hinge_reg(data=data, labels=labels, weights=weights_in)


def vanilla_grad_descent(loss, data_in, labels_in, weights_in, iterations=100, step_size=0.001):
    """
    Basic gradient descent.

    :param loss:
    :param data_in:
    :param labels_in:
    :param weights_in:
    :param iterations:
    :param step_size:
    :return:
    """
    weights_out = np.copy(weights_in)
    #print weights_out

    for i in range(0, iterations):
        weights_grad = loss(data_in, labels_in, weights_out)
        weights_out += step_size*weights_grad
        #print weights_grad, weights_out, "\n"
    return weights_out


def vanilla_grad_descent_num(loss, weights_in, iterations=100, step_size=0.001):
    """
    Basic gradient descent using numerical gradient.

    :param loss:
    :param weights_in:
    :param iterations:
    :param step_size:
    :return:
    """
    weights_out = np.copy(weights_in)
    #print weights_out

    for i in range(0, iterations):
        weights_grad = gradients.eval_numerical_gradient(loss, weights_in)
        weights_out += step_size*weights_grad
        #print weights_grad, weights_out, "\n"
    return weights_out


# set seed
np.random.seed(0)

nsamp = 100
data = np.vstack((np.linspace(-5, 5, nsamp), np.ones((nsamp,))))
labels = np.hstack((np.zeros((40,)),
                    np.ones((60,))
                    ))
labels = labels.astype(int)
ndim = data.shape[0]
nclasses = np.unique(labels).shape[0]
weights = np.random.randn(nclasses, ndim)
preds = lin_class(data, weights)

# compute loss
print "Weight shape: " + str(weights.shape)
print "Data shape: " + str(data.shape)
print "Labels shape: " + str(labels.shape)
loss_val = losses.loss_hinge_reg(data, labels, weights)
print "Computed loss: " + str(loss_val)

# compute analytical and numerical gradients
grada = gradients.hinge_loss_grad(data, labels, weights)
grada_slow = gradients.hinge_loss_grad_slow(data, labels, weights)
gradn = gradients.eval_numerical_gradient_twoside(data_loss_hinge, weights)

print "Analytical gradient (fast): \n" + str(grada)
print "Analytical gradient (slow): \n" + str(grada_slow)
print "Numerical gradient: \n" + str(gradn)

# compute ideal gradients
print "Analytical gradient stream: "
weights_a = vanilla_grad_descent(gradients.hinge_loss_grad, data_in=data, labels_in=labels, weights_in=weights,
                                 iterations=1000, step_size=0.05)
print "Numerical gradient stream: "
weights_num = vanilla_grad_descent_num(data_loss_hinge, weights_in=weights, iterations=1000, step_size=0.05)

inds1 = np.nonzero(labels == 0)[0]
inds2 = np.nonzero(labels == 1)[0]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(data[0, inds1], labels[inds1], 'ro', markersize=5.0, label='class 1')
ax.plot(data[0, inds2], labels[inds2], 'go', markersize=5.0, label='class 2')
ax.plot(data[0, :], preds[0, :], 'r-', linewidth=3.0, label='classifier 1')
ax.plot(data[0, :], preds[1, :], 'g-', linewidth=3.0, label='classifier 2')
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.title.set_text("Linear classifier (initial)")
ax.legend()

preds_a = lin_class(data, weights=weights_a)
preds_num = lin_class(data, weights=weights_num)

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(data[0, inds1], labels[inds1], 'ro', markersize=5.0, label='class 1')
ax2.plot(data[0, inds2], labels[inds2], 'go', markersize=5.0, label='class 2')
ax2.plot(data[0, :], preds_a[0, :], 'r-', linewidth=3.0, label='classifier 1')
ax2.plot(data[0, :], preds_a[1, :], 'g-', linewidth=3.0, label='classifier 2')
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")

ax2.title.set_text("Linear classifier (analytical gradient)")
ax2.legend()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(data[0, inds1], labels[inds1], 'ro', markersize=5.0, label='class 1')
ax3.plot(data[0, inds2], labels[inds2], 'go', markersize=5.0, label='class 2')
ax3.plot(data[0, :], preds_num[0, :], 'r-', linewidth=3.0, label='classifier 1')
ax3.plot(data[0, :], preds_num[1, :], 'g-', linewidth=3.0, label='classifier 2')
ax3.set_xlabel("x")
ax3.set_ylabel("f(x)")

ax3.title.set_text("Linear classifier (numerical gradient)")
ax3.legend()


plt.show()
