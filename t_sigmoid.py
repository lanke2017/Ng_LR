import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


if __name__ == '__main__':

    a = np.array([-0.403])
    a = a + 512
    a = a / 5000
    a = sigmoid(a)
    a = a * 1024
    

    print(a)
