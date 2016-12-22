import os
import numpy as np
import matplotlib.pyplot as plt
from dnncode.utils import unpickle, gen_image

data_dir = "../datasets/cifar-10-batches-py/"
data_file1 = "data_batch_1"
batch1_file = os.path.abspath(os.path.join(data_dir, data_file1))

dict1 = unpickle(batch1_file)
data = dict1['data']
labels = np.array(dict1['labels'])

# visualize series of images
seed = 0
nrows = 20
ncols = 40
width = 32
height = 32

# set random seed, and generate random images
nsamp = data.shape[0]
np.random.seed(seed=seed)
data_keep = data[np.random.permutation(nsamp)[0:nrows*ncols], :]

row_imgs = []
rows = []
for i in range(0, nrows):
    imgs = data_keep[i*ncols:(i+1)*ncols, :]
    for j in range(0, ncols):
        row_imgs.append(gen_image(imgs[j, :], width, height))
    rows.append(np.hstack(row_imgs))
    row_imgs = []

full_img = np.vstack(rows)
plt.imshow(full_img)
plt.title("Random images from CIFAR-10: " + str(nrows) + "x" + str(ncols) + " grid")
plt.show()
