import numpy as np
from utility import make_samples_gauss
import pylab as pl

dim = 4
mu0 = -0.25 * np.ones(dim)
sigma = 2 * np.eye(dim)
n = 100

mu1 = 0.25 * np.ones(dim)

res0 = make_samples_gauss(mu0, sigma, n, dim, random_state=0)

res1 = make_samples_gauss(mu1, sigma, n, dim, random_state=1)

print(sum(res1[:, 0])/n)

pl.figure(1)
pl.scatter(res0[:, 0], res0[:, 1])
pl.show()
