import numpy as np
from utility import make_samples_gauss, gradcheck_naive, logistic_loss, \
    cost_function, gradient_update, gradient_decent, \
    stochastic_grad_decent, hypercube_proj, ball_proj, make_random_labels
import pylab as pl

dim = 4
mu0 = -0.25 * np.ones(dim)
sigma = 2 * np.eye(dim)
n = 100

mu1 = 0.25 * np.ones(dim)
samples = np.zeros((n, dim))
labels = make_random_labels(n)
num_neg = len(labels[labels == -1])
num_pos = n - num_neg
samples_pos = make_samples_gauss(mu0, sigma, num_pos, dim, random_state=0)
samples_neg = make_samples_gauss(mu1, sigma, num_neg, dim, random_state=1)
samples[labels == 1] = hypercube_proj(samples_pos)
samples[labels == -1] = hypercube_proj(samples_neg)
print(samples)
# t_X = hypercube_proj(t_X)
# quad = lambda x: (np.sum(x ** 2), x * 2)
# w_hat = stochastic_grad_decent(random_w, t_X, t_y, alpha=0.1, max_iterations=T)
