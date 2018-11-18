import numpy as np
from utility import make_samples_gauss, gradcheck_naive, logistic_loss, \
    cost_function
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
gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test

print(logistic_loss(1))
t_X = np.array([[1, 2], [-1, -2]])
t_y = np.array([-1, 1])
t_theta1 = np.array([-10, 10])
t_theta2 = np.array([10, -10])
t_c1 = cost_function(t_theta1, t_X, t_y)
t_c2 = cost_function(t_theta2, t_X, t_y)
print ("=== For autograder ===")
print (t_c1)
print (t_c2)
