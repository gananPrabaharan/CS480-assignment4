import numpy as np


def kernel(s, t, sigma, part_der_flag=False):
    s = np.array(s)
    t = np.array(t)

    if part_der_flag:
        return (sigma-np.power(s, 2)+2*np.matmul(s,t)-np.power(t, 2))/sigma*np.exp(-np.power(s-t,2)/(2*sigma))

    return np.exp(-(np.power((s-t), 2)/(2*sigma)))


