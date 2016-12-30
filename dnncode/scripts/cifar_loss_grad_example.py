import time
import os
import numpy as np
import matplotlib.pyplot as plt
from dnncode.utils import unpickle, gen_image
from dnncode.classifiers import losses, gradients


def CIFAR_loss_hinge_fun(W):
    """
    Compute CIFAR loss using hinge loss.

    :param W: weight matrix
    :return:
    """
    return losses.loss_hinge_reg(data=data.T, labels=labels, weights=W)


def CIFAR_loss_hinge_part(W):
    """
    Compute CIFAR loss using hinge loss.

    :param W: weight matrix
    :return:
    """
    curr_loss = 0.0
    for i in range(0, labels.shape[0]):
        curr_loss += losses.loss_hinge_partvec(x=data[i, :], y=labels[i], W=W)
    return curr_loss/float(labels.shape[0])

subsample = True
compute_numerical = False
data_dir = "../datasets/cifar-10-batches-py/"
data_file = "data_batch_" + str(1)
batch_file = os.path.abspath(os.path.join(data_dir, data_file))
dict_dat = unpickle(batch_file)
data = dict_dat['data']
labels = np.array(dict_dat['labels'])

# set seed and compute loss gradient (painfully slow)
np.random.seed(0)

if subsample:
    nsamp = data.shape[0]
    nret = int(0.05*nsamp)
    rand_inds = np.random.permutation(nsamp)[0:nret]
    data = data[rand_inds, :]
    labels = labels[rand_inds]

W = np.random.rand(10, 3072)*0.001  # random weight matrix
print data.shape, W.shape

s_t = time.time()
dfa = gradients.hinge_loss_grad(data=data.T, labels=labels, weights=W)
print "Analytical gradient computation time over %f dimensional vector using " \
      "10,000 samples: %f seconds." % (10*3072, time.time() - s_t)
loss_original = CIFAR_loss_hinge_fun(W)

if compute_numerical:
    s_t = time.time()
    df = gradients.eval_numerical_gradient(CIFAR_loss_hinge_fun, W)
    print "Numerical gradient computation time over %f dimensional vector using " \
          "10,000 samples: %f seconds." % (10*3072, time.time() - s_t)
    print "Difference between analytical and numerical gradient: " + str(np.linalg.norm(df-dfa))

    print "Computing numerical gradient steps..."
    print "Original loss: %f" % (loss_original, )

    s_t = time.time()
    for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
        step_size = float(10) ** step_size_log
        W_new = W - step_size*df
        loss_new = CIFAR_loss_hinge_fun(W_new)
        print "For step size %f new loss: %f" % (step_size, loss_new)
    print "Loss computation time over 10 steps: " + str(time.time()-s_t) + " seconds."

print "Computing analytical gradient steps..."
print "Original loss: %f" % (loss_original, )

s_t = time.time()
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = float(10) ** step_size_log
    W_new = W - step_size*dfa
    loss_new = CIFAR_loss_hinge_fun(W_new)
    print "For step size %f new loss: %f" % (step_size, loss_new)
print "Loss computation time over 10 steps: " + str(time.time()-s_t) + " seconds."


# now perform gradient check
gradients.grad_check_sparse(f=CIFAR_loss_hinge_fun, x=W, analytic_grad=dfa, num_checks=1000)


# visualize gradients
nrows = 2
ncols = 5
row_imgs = []
rows = []
for i in range(0, nrows):
    imgs = dfa[i*ncols:(i+1)*ncols, :]
    for j in range(0, ncols):
        row_imgs.append(gen_image(imgs[j, :]))
    rows.append(np.hstack(row_imgs))
    row_imgs = []

full_img = np.vstack(rows)
plt.figure()
plt.imshow(full_img)
plt.title("Analytical gradient vector on CIFAR-10: " + str(nrows) + "x" + str(ncols) + " grid")

if compute_numerical:
    row_imgs = []
    rows = []
    for i in range(0, nrows):
        imgs = df[i*ncols:(i+1)*ncols, :]
        for j in range(0, ncols):
            row_imgs.append(gen_image(imgs[j, :]))
        rows.append(np.hstack(row_imgs))
        row_imgs = []

    full_img = np.vstack(rows)
    plt.figure()
    plt.imshow(full_img)
    plt.title("Numerical gradient vector on CIFAR-10: " + str(nrows) + "x" + str(ncols) + " grid")

plt.show()
