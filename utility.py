import numpy as np
import scipy as sp
from scipy import linalg
import random

def cost_function(w, X, y):

    z = y * np.matmul(X, w)
    cost = np.sum(logistic_loss(z))

    return cost


def logistic_loss(z):

    s = np.log(1 + np.exp(-z))

    return s

def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()

    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point

#     print(x)
#     print(fx)
#     print(grad)

    h = 1e-4
#     print('x: ')

#     print(f(x))
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index

        thetaPlus = np.array(x)
        thetaPlus[ix] += h

        thetaMinus = np.array(x)
        thetaMinus[ix] -= h

        ### random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later

        rndstate = random.getstate()
        random.setstate(rndstate)

        jthetaP, tmp = f(thetaPlus)

        rndstate = random.getstate()
        random.setstate(rndstate)

        jthetaM, tmp = f(thetaMinus)
        numgrad = (jthetaP - jthetaM) / (2 * h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print ("Gradient check failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print ("Gradient check passed!")

# sigma: covariance matrix
def make_samples_gauss(mu, sigma, n, dim, random_state=None):

    generator = check_random_state(random_state)

    if np.isscalar(sigma):
        sigma = np.array([sigma, ])
    if len(sigma) > 1:
        p = sp.linalg.sqrtm(sigma)
        res = generator.randn(n, dim).dot(p) + mu
    else:
        print('sigma should be a matrix, not a scalar')
        res = 0
    return res

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))
