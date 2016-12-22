"""
Utilities for implementing deep neural networks.
"""
import cPickle
import numpy as np


def unpickle(file_in):
    """
    Unpickle a file and return a dictionary.

    :param file_in:
    :return:
    """
    fo = open(file_in, 'rb')
    dict_out = cPickle.load(fo)
    fo.close()
    return dict_out


def gen_image(im_vec, width=32, height=32):
    """
    Return a numpy array representing a color image from a vector.

    :param im_vec:
    :param width:
    :param height:
    :return:
    """
    return im_vec.reshape(3, width, height).transpose(1, 2, 0)

