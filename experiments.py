import numpy as np
from utility import make_samples_gauss, gradcheck_naive, logistic_loss, \
    cost_function, gradient_update, gradient_decent, \
    stochastic_grad_decent, hypercube_proj, ball_proj, \
    make_random_labels, predict, generate_samples
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.model_selection import train_test_split

dim = 4
mu0 = -0.25 * np.ones(dim)
mu1 = 0.25 * np.ones(dim)
vals = [0.05, 0.3]
filenames = ['005_', '03_']
for j in range(len(vals)):

    sigma = (vals[j]**2) * np.eye(dim)

    params = ['hypercube', 'ball']

    for iparam in range(len(params)):

        proj = params[iparam]
        print('---Generate test samples---')
        seed0, seed1 = 100, 101
        samples_test, labels_test = generate_samples(mu0, mu1, \
                sigma, sigma, dim, 400, seed0, seed1, proj=proj)
        print('---Done---')

        nsamples_ = [50, 100, 500, 1000]
        nruns = 30
        risks = np.zeros((len(nsamples_), nruns))
        clf_errors = np.zeros((len(nsamples_), nruns))

        for i in range(len(nsamples_)):

            nsamples = nsamples_[i]
            T = nsamples
            for irun in range(nruns):

                # seed0, seed1 = np.random.randint(100, size=1)[0], \
                #                 np.random.randint(100, size=1)[0]
                seed0 = irun
                seed1 = irun
                samples_train, labels_train = generate_samples(mu0, mu1, \
                        sigma, sigma, dim, nsamples, seed0, seed1, proj=proj)
                w_initial = np.zeros(dim+1)
                w_hat, cost = stochastic_grad_decent(w_initial, samples_train, labels_train, \
                        alpha=0.1, max_iterations=T, proj=proj)
                predicted_labels =  predict(w_hat, samples_test)
                t_precision = predicted_labels[np.where(predicted_labels == labels_test)].size / float(labels_test.size) * 100
                print('Accuracy on the test set: %s%%' % round(t_precision,2))
                nrow = samples_test.shape[0]
                prefix = np.repeat(1, nrow)
                testX = np.append(prefix[:, None], samples_test, axis = 1)
                risks[i][irun] = cost_function(w_hat, testX, labels_test)
                clf_errors[i][irun] = t_precision / 100

        excess_risks = np.zeros(len(nsamples_))
        std_excess_risks = np.zeros(len(nsamples_))
        mean_risks = np.mean(risks, axis=1)
        min_risks = np.min(risks, axis=1)
        excess_risks = mean_risks - min_risks
        std_excess_risks = np.std(risks, axis=1)
        mean_clf_errors = np.mean(clf_errors, axis=1)
        std_clf_errors = np.std(clf_errors, axis=1)
        pl.figure(j, (5, 4))
        pl.errorbar(nsamples_, excess_risks, yerr=std_excess_risks, fmt='-o')
        print(excess_risks)
        print(std_excess_risks)
        print(filenames[j], params[iparam])
        df = pd.DataFrame({'Mean_risk': min_risks, 'Std_risk': std_excess_risks, \
            'Min_risk': min_risks, 'Excess_risk': excess_risks, \
            'Mean_error': mean_clf_errors, 'Std_error': std_clf_errors})
        file = filenames[j] + params[iparam] + '.csv'
        df.to_csv(file, index=False, header=True)
        # np.savetxt(filenames[j] + params[iparam] + '.csv', \
        #     zip(mean_risks), \
        #     # , std_excess_risks), \
        #     # min_risks, excess_risks, mean_clf_errors, std_clf_errors), \
        #     delimiter=',')
        #     # header='Mean_risk, Std_risk, Min_risk, Excess_risk, Mean_error, Std_error')

pl.show()
