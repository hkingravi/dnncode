import numpy as np
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb


def gencolorarray(numcolors):
    """
    Generate an array of colors varying by hue.

    :param numcolors:
    :return:
    """
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


def trap_func(x_in, start_up, saturate, height, start_down, zero_out):
    """
    Return a trapeziod function based on given parameters.

    :param x_in:
    :param start_up:
    :param saturate:
    :param height:
    :param start_down:
    :param zero_out:
    :return:
    """
    y_out = np.zeros(x_in.shape)
    for i in range(0, x_in.shape[0]):
        if saturate > x_in[i] >= start_up:
            y_out[i] = (float(height)/(saturate-start_up))*x_in[i] - (float(height)/(saturate-start_up))*start_up
        elif start_down > x_in[i] >= saturate:
            y_out[i] = height
        elif zero_out > x_in[i] >= start_down:
            y_out[i] = -(float(height)/(zero_out-start_down))*x_in[i] +\
                       (float(height)/(zero_out-start_down))*start_down + height
    return y_out


def rand_param_gen(nfuncs, x_start, x_finish):
    """
    Generate random trapezoid functions.

    :param nfuncs:
    :param x_start:
    :param x_finish:
    :return:
    """
    param_mat = np.zeros((5, nfuncs))
    params = np.random.rand(4, nfuncs)

    for i in range(0, nfuncs):
        curr_vec = x_finish*np.sort(params[:, i]) - x_start
        rand_height = np.random.rand(1)
        param_mat[0, i] = curr_vec[0]
        param_mat[1, i] = curr_vec[1]
        param_mat[2, i] = rand_height
        param_mat[3, i] = curr_vec[2]
        param_mat[4, i] = curr_vec[3]

    return param_mat


x_start = 0
x_finish = 10
nsamp = 1000
x = np.linspace(x_start, x_finish, nsamp)
np.random.seed(0)

nfuncs = 10
params = rand_param_gen(nfuncs, x_start=0, x_finish=10)

y_arrs = []
for i in range(0, nfuncs):
    y_arrs.append(trap_func(x, start_up=params[0, i], saturate=params[1, i], height=params[2, i],
                            start_down=params[3, i], zero_out=params[4, i]))

colors = gencolorarray(len(y_arrs))
y = np.ones(y_arrs[0].shape)
for curr_vec in y_arrs:
    y = np.multiply(y, 1 - curr_vec)
y = 1 - y


plt.figure()
for i, curr_y in enumerate(y_arrs):
    label_val = "trap " + str(i)
    plt.plot(x, curr_y, color=colors[i], label=label_val)
plt.title("Collection of trapezoid functions")

plt.figure()
plt.plot(x, y)
plt.title("Action of product function")
plt.show()
