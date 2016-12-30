import time
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from dnncode.utils import unpickle
from dnncode.classifiers import losses

data_dir = "../datasets/cifar-10-batches-py/"
data_full = os.path.abspath(os.path.join(data_dir, "cifar_full.pkl"))

# set seed
np.random.seed(0)

# load data
print "Loading data from cached file " + data_full + "..."
load_t = time.time()
data_full_dict = pickle.load(open(data_full, "rb"))
data = data_full_dict['data']
labels = data_full_dict['labels']
print "Time taken to load data: " + str(time.time()-load_t)

# make up weight matrix to test
nsamp = data.shape[0]
data = np.hstack((data, np.ones((nsamp, 1))))  # add bias vector to data
dim = data.shape[1]
nclasses = np.unique(labels).shape[0]
weights = np.random.randn(nclasses, dim)

# time loss computations
unvec_s = time.time()
losses_unvec = np.zeros((nsamp,))
for i in range(0, nsamp):
    losses_unvec[i] = losses.loss_hinge_unvec(data[i, :], labels[i], weights)
print "Time taken to run unvectorized hinge loss on " + str(dim) \
      + " dimensional data with " + str(nsamp) + " samples: " \
      + str(time.time() - unvec_s) + " seconds."

partvec_s = time.time()
losses_partvec = np.zeros((nsamp,))
for i in range(0, nsamp):
    losses_partvec[i] = losses.loss_hinge_partvec(data[i, :], labels[i], weights)
print "Time taken to run partially vectorized hinge loss on " + str(dim) \
      + " dimensional data with " + str(nsamp) + " samples: " \
      + str(time.time() - partvec_s) + " seconds."

fullvec_s = time.time()
losses_fullvec = losses.loss_hinge(data, labels, weights)
print "Time taken to run fully vectorized hinge loss on " + str(dim) \
      + " dimensional data with " + str(nsamp) + " samples: " \
      + str(time.time() - fullvec_s) + " seconds."

fullvec_reg_s = time.time()
losses_fullvec_reg = losses.loss_hinge_reg(data, labels, weights, reg_val=0.1)
print "Time taken to run regularized fully vectorized hinge loss on " + str(dim) \
      + " dimensional data with " + str(nsamp) + " samples: " \
      + str(time.time() - fullvec_reg_s) + " seconds."

print "Unvectorized loss: " \
      + str(np.sum(losses_unvec)) + "."
print "Partially vectorized loss: " \
      + str(np.sum(losses_partvec)) + "."
print "Fully vectorized loss: " \
      + str(losses_fullvec) + "."
print "Fully vectorized regularized loss: " \
      + str(losses_fullvec_reg) + "."


# softmax section
fullvec_reg_s = time.time()
losses_soft = losses.loss_softmax(data, labels, weights)
print "Time taken to run regularized fully vectorized softmax loss on " + str(dim) \
      + " dimensional data with " + str(nsamp) + " samples: " \
      + str(time.time() - fullvec_reg_s) + " seconds."
print "Fully vectorized regularized softmax loss: " \
      + str(losses_soft) + "."

