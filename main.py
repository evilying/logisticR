import numpy as np
from utility import make_samples_gauss, gradcheck_naive, logistic_loss, \
    cost_function, gradient_update
import pylab as pl

dim = 4
mu0 = -0.25 * np.ones(dim)
sigma = 2 * np.eye(dim)
n = 100

mu1 = 0.25 * np.ones(dim)

res0 = make_samples_gauss(mu0, sigma, n, dim, random_state=0)

res1 = make_samples_gauss(mu1, sigma, n, dim, random_state=1)

quad = lambda x: (np.sum(x ** 2), x * 2)

gradcheck_naive(quad, np.array(123.456))      # scalar test

t_samples, t_features = 100, 10
t_X = np.random.randn(t_samples, t_features)
t_y = np.ones(t_samples)
t_y[: int(t_samples/2)] *= -1

random_theta = np.random.randn(t_features)

gradcheck_naive(lambda t_theta: (cost_function(t_theta, t_X, t_y), \
    gradient_update(t_theta, t_X, t_y)), random_theta)
