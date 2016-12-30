"""
Utilities for implementing deep neural networks.
"""
import cPickle
from colorsys import hsv_to_rgb


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


def gencolorarray(numcolors):
    # ensure numcolors is an integer by using exception
    color_list = []
    try:
        for i in xrange(1, numcolors + 1):
            p_color = float(i) / numcolors
            color_val = hsv_to_rgb(p_color, 1, 1)
            color_list.append(color_val)
    except:
        print "numcolors must be an integer\n"

    return color_list
