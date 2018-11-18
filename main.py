import numpy as np
from utility import make_samples_gauss, gradcheck_naive, logistic_loss, \
    cost_function, gradient_update, gradient_decent, \
    stochastic_grad_decent, hypercube_proj, ball_proj, \
    make_random_labels, predict
import pylab as pl
from sklearn.model_selection import train_test_split

dim = 4
mu0 = -0.25 * np.ones(dim)
sigma = 2 * np.eye(dim)
n = 1000

mu1 = 0.25 * np.ones(dim)
samples = np.zeros((n, dim))
labels = make_random_labels(n)
num_neg = len(labels[labels == -1])
num_pos = n - num_neg
samples_pos = make_samples_gauss(mu0, sigma, num_pos, dim, random_state=2)
samples_neg = make_samples_gauss(mu1, sigma, num_neg, dim, random_state=1)

# params = ['hypercube', 'ball']
params = ['hypercube']

for i in range(len(params)):

    print('-----scenario', i+1, '-----')
    param = params[i]
    if param == 'hypercube':
        samples[labels == 1] = hypercube_proj(samples_pos)
        samples[labels == -1] = hypercube_proj(samples_neg)
    else:
        samples[labels == 1] = ball_proj(samples_pos)
        samples[labels == -1] = ball_proj(samples_neg)

    samples_train, samples_test, labels_train, labels_test = \
        train_test_split(samples, labels, test_size=400, random_state=0)
    w_initial = np.zeros(dim+1)

    T = 1000
    w_hat = stochastic_grad_decent(w_initial, samples_train, labels_train, \
            alpha=0.01, max_iterations=T)
    print(w_hat)
    predicted_labels =  predict(w_hat, samples_train)
    # print(predicted_labels)
    # print(labels_train)
    t_precision = predicted_labels[np.where(predicted_labels == labels_train)].size / float(labels_train.size) * 100

    print('Accuracy on the training set: %s%%' % round(t_precision,2))
