"""To compute the natural logarithm of a number, create/remove a directory (folder),
parameterise probability distributions, create static, animated, and interactive visualisations,
work with arrays, and carry out fast numerical computations in Python."""
from math import log
import os
import torch
import torch.distributions as dis
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
torch.manual_seed(12)
tf.random.set_seed(638)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = dis.Laplace(loc=0.1, scale=1)
q = dis.Normal(loc=0, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# parameters for the addition of Laplace and Gaussian noise
EPS = 0.1
DTA = 0.1
A = 0
b1 = log(2)
b2 = 2*((log(1.25))/DTA)*b1

# noise distributions
noise1 = tfp.distributions.Laplace(loc=A, scale=b1)
noise2 = tfp.distributions.Normal(loc=A, scale=b2)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# use multi-dimensional numpy arrays to save sampled points and statistics
qCS = np.zeros((C, N))
qOCS = np.zeros((C, N))
qT1 = np.zeros(C)
qT2 = np.zeros(C)
qOrderedT1 = np.zeros(C)
qOrderedT2 = np.zeros(C)
logr = np.zeros(C)
oLr = np.zeros(C)
k3 = np.zeros(C)
oK3 = np.zeros(C)

# store T before C (clients)
F = 5_000
Tset = np.array([1*F, 2*F, 5*F, 10*F, 20*F, 50*F, 100*F, 200*F, 500*F, 1_000*F])
L = np.size(Tset)
KLDest1 = np.zeros((C, L))
KLDest2 = np.zeros((C, L))
oKLDest1 = np.zeros((C, L))
oKLDest2 = np.zeros((C, L))
T_COUNT = 0

for T in Tset:

    # create separate numpy arrays for sampled and ordered cases
    k3noise1 = np.zeros(L)
    k3noise2 = np.zeros(L)
    oK3noise1 = np.zeros(L)
    oK3noise2 = np.zeros(L)

    # carry out 10 repeats
    for k in range(0, 10):

        # round to 2 d.p., find indices of and eliminate unique values
        qSample = q.sample(sample_shape=(T,))
        qRound = torch.round(qSample, decimals=2)
        qUnique = torch.unique(qRound, return_counts=True)
        qIndices = (qUnique[1] == 1).nonzero().flatten()
        qUniqueIndices = qUnique[0][qIndices]

        for i in qUniqueIndices:
            qRound = qRound[qRound != i]

        # order the pre-processed sample and separate into + and - values
        qOrderedRound = torch.sort(qRound)
        qNegativeRound = qOrderedRound[0][0:2440]
        qPositiveRound = qOrderedRound[0][2440:4918]

        for j in range(0, C):

            # even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                indices = torch.randperm(len(qPositiveRound))[:N]
                qCS[j] = qPositiveRound[indices]
            else:
                indices = torch.randperm(len(qNegativeRound))[:N]
                qCS[j] = qNegativeRound[indices]

            qT1[j] = torch.numel(torch.from_numpy(qCS[j]))
            qCS[j] = qCS[j][abs(int(qT1 - 0.98*T)):]
            qT2[j] = torch.numel(torch.from_numpy(qCS[j]))

            # each client gets 500 points in order from ordered pre-processed sample
            qOCS[j] = qOrderedRound[0][(500*j):(500*(j+1))]
            qOrderedT1[j] = torch.numel(torch.from_numpy(qOCS[j]))
            qOCS[j] = qOCS[j][abs(int(qOrderedT1 - 0.98*T)):]
            qT2[j] = torch.numel(torch.from_numpy(qOCS[j]))

            # skip Prio step for now
            logr[j] = (p.log_prob(torch.from_numpy(qCS[j])) - q.log_prob(torch.from_numpy(qCS[j])))
            oLr[j] = (p.log_prob(torch.from_numpy(qOCS[j])) - q.log_prob(torch.from_numpy(qOCS[j])))
            k3[j] = (logr[j].exp() - 1) - logr[j]
            oK3[j] = (oLr[j].exp() - 1) - oLr[j]

            # add Laplace and Gaussian noise with parameter(s) eps (and dta)
            k3noise1[k] = k3 + (noise1.sample(sample_shape=qT2[j], seed=12)/EPS)
            k3noise2[k] = k3 + (noise2.sample(sample_shape=qT2[j], seed=12)/EPS)
            oK3noise1[k] = oK3 + (noise1.sample(sample_shape=qOrderedT2[j], seed=12)/EPS)
            oK3noise2[k] = oK3 + (noise2.sample(sample_shape=qOrderedT2[j], seed=12)/EPS)

        average1 = np.sum(k3noise1) / np.size(k3noise1)
        average2 = np.sum(k3noise2) / np.size(k3noise2)
        oAverage1 = np.sum(oK3noise1) / np.size(oK3noise1)
        oAverage2 = np.sum(oK3noise2) / np.size(oK3noise2)

        # compare with true KLD
        KLDest1[C, T_COUNT] = abs(np.mean(average1) - truekl)
        KLDest2[C, T_COUNT] = abs(np.mean(average2) - truekl)
        oKLDest1[C, T_COUNT] = abs(np.mean(oAverage1) - truekl)
        oKLDest2[C, T_COUNT] = abs(np.mean(oAverage2) - truekl)
        T_COUNT = T_COUNT + 1

# compute mean of KLD for particular T across all clients
KLDmean1 = np.mean(KLDest1, axis = 0)
KLDmean2 = np.mean(KLDest2, axis = 0)
oKLDmean1 = np.mean(oKLDest1, axis = 0)
oKLDmean2 = np.mean(oKLDest2, axis = 0)

plot1 = plt.plot(Tset, KLDest1, label = "Laplace (sampled)")
plot2 = plt.plot(Tset, KLDest2, label = "Gaussian (sampled)")
plot3 = plt.plot(Tset, oKLDest1, label = "Laplace (ordered)")
plot4 = plt.plot(Tset, oKLDest2, label = "Gaussian (ordered)")

plt.title("Effect of T on noisy estimate of KLD")
plt.xlabel("Value of T")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()
