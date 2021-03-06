"""
Implementations of various gradients useful for DNNs.
"""
import numpy as np
from random import randrange


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a blob
    into which outputs will be written. For example, f might be called like this:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


def eval_numerical_gradient_orig(f, x, verbose=True, h=0.00001):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print ix, grad[ix]
    it.iternext() # step to next dimension

  return grad

def eval_numerical_gradient(f, x, h=1e-5):
    """
    Naive implementation of numerical gradient of f at x.
    :param f: function taking in a single argument
    :param x: point to evaluate gradient at (numpy array)
    :param h: 'gridding' parameter
    :return:
    """
    fx = f(x)
    grad = np.zeros(x.shape)

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    curr_it = 0
    while not it.finished:
        # eval function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(x)  # evaluate f(x+h)
        x[ix] = old_value  # restore to previous value (remember, everything is by reference)

        # compute partial derivative
        grad[ix] = (fxh - fx)/h  # slope
        it.iternext()
        curr_it += 1

    return grad


def eval_numerical_gradient_twoside(f, x, h=1e-5):
    """
    Two-sided implementation of numerical gradient of f at x.
    :param f: function taking in a single argument
    :param x: point to evaluate gradient at (numpy array)
    :param h: 'gridding' parameter
    :return:
    """
    grad = np.zeros(x.shape)

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # eval function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh_f = f(x)  # evaluate f(x+h)
        x[ix] = old_value - h  # restore to previous value (remember, everything is by reference)
        fxh_b = f(x)
        x[ix] = old_value

        # compute partial derivative
        grad[ix] = (fxh_f - fxh_b)/(2*h)  # slope
        it.iternext()

    return grad


def hinge_loss_grad_slow(data, labels, weights):
    """
    Compute hinge-loss gradient analytically. Involves looping, multiple times. Slow as hell.

    :param data: holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    :param labels: array of integers specifying correct class (e.g. 50,000 array)
    :param weights: are weights (e.g. 10 x 3073)
    :return:
    """
    delta = 1.0
    nsamp = labels.shape[0]
    nclasses = weights.shape[0]
    ndim = weights.shape[1]
    scores = weights.dot(data)  # compute dot product

    # need to construct the gradient
    grad_mat = np.zeros((nclasses, nsamp))
    grad_mat_cache = np.zeros((nclasses, nsamp))

    for i in range(0, nsamp):
        for j in range(0, nclasses):
            # go through all classes that are not the actual class, and get values
            if j != labels[i]:
                grad_mat_cache[j, i] = scores[j, i] - scores[labels[i], i] + delta
                if grad_mat_cache[j, i] > 0:
                    grad_mat[j, i] = 1
        # once you've looped through, now compute sum
        grad_mat[labels[i], i] = -np.sum(grad_mat[:, i])

    # now go along each dimension and compute overall loss
    curr_grads = []
    for j in range(0, nclasses):
        curr_vec = np.zeros((ndim,))
        for i in range(0, nsamp):
            curr_vec += grad_mat[j, i]*data[:, i]

        curr_grads.append(curr_vec/float(nsamp))
    return np.vstack(curr_grads)


def hinge_loss_grad(data, labels, weights, reg_val=0.0):
    """
    Compute hinge-loss gradient analytically. See notes on derivation and
    numpy tricks utilized to speed up compute time (broadcasting, and in-place
    referencing).

    :param data: holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    :param labels: array of integers specifying correct class (e.g. 50,000 array)
    :param weights: are weights (e.g. 10 x 3073)
    :param reg_val: regularization value
    :return:
    """
    delta = 1.0
    nsamp = labels.shape[0]
    indexer = np.arange(nsamp)
    scores = weights.dot(data)  # compute dot product
    correct_scores = scores[labels, indexer]
    grad = scores - correct_scores + delta
    grad[np.nonzero(grad < 0)] = 0
    grad[labels, indexer] = 0  # neutralize correct labels
    grad[np.nonzero(grad > 0)] = 1
    grad[labels, indexer] = -np.sum(grad, axis=0)  # assign summed values
    grad = (1/float(nsamp))*data.dot(grad.T) + 2*reg_val*weights.T
    return grad.T


def softmax_loss_grad(data, labels, weights, reg_val=0.0):
    """
    Compute softmax loss gradient analytically.

    :param data:
    :param labels:
    :param weights:
    :param reg_val:
    :return:
    """


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """
    for i in xrange(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

    oldval = x[ix]
    x[ix] = oldval + h  # increment by h
    fxph = f(x)  # evaluate f(x + h)
    x[ix] = oldval - h  # increment by h
    fxmh = f(x)  # evaluate f(x - h)
    x[ix] = oldval  # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
