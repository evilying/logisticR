import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy import linalg
import random
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def predict(w, X):

    X = np.array(X)
    nrow, ncol = X.shape
    prefix = np.repeat(1, nrow)

    valX = np.append(prefix[:, None], X, axis = 1)

    t = np.dot(valX, w)

    probabilities = np.exp(t) / (1 + np.exp(t))
    predicted_labels = (t > 0) * 1
    predicted_labels[t < 0] = -1

    return predicted_labels

def ball_proj(X):

    X_ball = X / la.norm(X, axis=1).reshape((X.shape[0], 1))

    return X_ball

def hypercube_proj(X):

    X_hat = np.zeros(X.shape)
    for i in range(len(X)):

        X_hat[i] = np.array(vec_hypercube_proj(X[i])).reshape((1, X.shape[1]))
    return X_hat

def vec_hypercube_proj(v):

    dim = len(v)
    I = matrix(np.eye(dim), (dim, dim), 'd')
    Q = 2 * I
    v_p = -2 * v
    p = matrix(v_p, (dim, 1), 'd')
    G = matrix([[I, -1 * I]])
    h = matrix(np.ones(2*dim), (2*dim, 1), 'd')
    sol=solvers.qp(Q, p, G, h)

    return sol['x']

def stochastic_grad_decent(w, X, y, alpha=1e-2, max_iterations=400, proj='hypercube'):

    scaleX = X / np.max(abs(X), 0)
    nrow, ncol = X.shape
    prefix = np.repeat(1, nrow)

    valX = np.append(prefix[:, None], scaleX, axis = 1)
    iteration = 0
    weights = np.zeros((max_iterations, len(w)))
    weights[0] = w
    while(iteration < max_iterations-1):

        iteration += 1
        ind_x_rand = np.random.randint(nrow, size=1)
        x_rand = valX[ind_x_rand]
        y_rand = y[ind_x_rand]
        grad = gradient_update(w, x_rand, y_rand)
        if proj == 'hypercube':
            grad_proj = np.array(vec_hypercube_proj(grad)).reshape((1, len(w)))[0]
            w = w.reshape((1, len(w)))[0]
            w -= alpha * grad_proj
        elif proj == 'ball':
            w -= alpha * ball_proj(grad)
        weights[iteration] = w

        w_hat = 1 / (iteration+1) * np.sum(weights, axis=0)
        cost = cost_function(w_hat, valX, y)
        print("[ Iteration", iteration, "]", "cost =", cost)

    if iteration == 0:
        print('gradient update does not occur!')
    w_hat = 1 / (iteration+1) * np.sum(weights, axis=0)

    return w_hat

def gradient_decent(w, X, y, alpha=1e-2, max_iterations=400):

    scaleX = X / np.max(abs(X), 0)
    nrow, ncol = X.shape
    prefix = np.repeat(1, nrow)

    valX = np.append(prefix[:, None], scaleX, axis = 1)

    iteration = 0
    while(iteration < max_iterations):

        iteration += 1
        w -= alpha * gradient_update(w, valX, y)
        cost = cost_function(w, valX, y)
        print("[ Iteration", iteration, "]", "cost =", cost)

def gradient_update(w, X, y):

    z = y * np.matmul(X, w)
    s = (-1 + 1 / (1 + np.exp(-z))) * y

    grad = np.sum(X * s[:, None], 0)
    grad /= X.shape[0]

    return grad

def cost_function(w, X, y):

    z = y * np.matmul(X, w)
    cost = np.sum(logistic_loss(z))

    cost /= X.shape[0]

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

def make_random_labels(n):

    labels = np.random.randint(2, size=n)
    labels[labels == 0] = -1

    return labels

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
