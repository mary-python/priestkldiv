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

# sample T points
T = 500_000

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
qPositiveRound = torch.where(qRound >= 0, qRound, -qRound)
qNegativeRound = torch.where(qRound < 0, qRound, -qRound)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# use multi-dimensional numpy arrays to save sampled points and statistics
qClSa = np.zeros((C, N))
qOrClSa = np.zeros((C, N))
qT = np.zeros(C)
qOrderedT = np.zeros(C)
logr = np.zeros(C)
oLogr = np.zeros(C)
k3 = np.zeros(C)
oK3 = np.zeros(C)

# parameters for the addition of Laplace and Gaussian noise
epsset = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4])
DTA = 0.1
A = 0
b1 = log(2)
b2 = 2*((log(1.25))/DTA)*b1

# noise distributions and multi-dimensional numpy arrays to store KLD estimates
noise1 = tfp.distributions.Laplace(loc=A, scale=b1)
noise2 = tfp.distributions.Normal(loc=A, scale=b2)
L = np.size(epsset)
KLDest1 = np.zeros((C, L))
KLDest2 = np.zeros((C, L))
oKLDest1 = np.zeros((C, L))
oKLDest2 = np.zeros((C, L))

for j in range(0, C):

    # even clients get positive values, odd clients get negative values
    if (j % 2) == 0:
        qClSa[j] = torch.multinomial(qPositiveRound, N, False)
    else:
        qClSa[j] = -torch.multinomial(qNegativeRound, N, False)

    qT[j] = torch.numel(torch.from_numpy(qClSa[j]))

    # each client gets 500 points in order from ordered pre-processed sample
    qOrClSa[j] = qOrderedRound[0][500*j:500*(j+1)]
    qOrderedT[j] = torch.numel(torch.from_numpy(qOrClSa[j]))

    # skip Prio step for now
    logr[j] = (p.log_prob(torch.from_numpy(qClSa[j])) - q.log_prob(torch.from_numpy(qClSa[j])))
    oLogr[j] = (p.log_prob(torch.from_numpy(qOrClSa[j])) - q.log_prob(torch.from_numpy(qOrClSa[j])))
    oK3[j] = (oLogr[j].exp() - 1) - oLogr[j]
    EPS_COUNT = 0

    for eps in epsset:

        # create separate numpy arrays for sampled and ordered cases
        k3noise1 = np.zeros(L)
        k3noise2 = np.zeros(L)
        oK3noise1 = np.zeros(L)
        oK3noise2 = np.zeros(L)

        # add average of 10 possible noise terms
        for k in range(0, 10):
            k3noise1[k] = k3 + (noise1.sample(sample_shape=qT[j], seed=12))/eps
            k3noise2[k] = k3 + (noise2.sample(sample_shape=qT[j], seed=12))/eps
            oK3noise1[k] = oK3 + (noise1.sample(sample_shape=qOrderedT[j], seed=12))/eps
            oK3noise2[k] = oK3 + (noise2.sample(sample_shape=qOrderedT[j], seed=12))/eps

        average1 = np.sum(k3noise1) / np.size(k3noise1)
        average2 = np.sum(k3noise2) / np.size(k3noise2)
        oAverage1 = np.sum(oK3noise1) / np.size(oK3noise1)
        oAverage2 = np.sum(oK3noise2) / np.size(oK3noise2)

        # compare with true KLD
        KLDest1[C, EPS_COUNT] = abs(np.mean(average1) - truekl)
        KLDest2[C, EPS_COUNT] = abs(np.mean(average2) - truekl)
        oKLDest1[C, EPS_COUNT] = abs(np.mean(oAverage1) - truekl)
        oKLDest2[C, EPS_COUNT] = abs(np.mean(oAverage2) - truekl)
        EPS_COUNT = EPS_COUNT + 1

# compute mean of KLD for particular epsilon across all clients
KLDmean1 = np.mean(KLDest1, axis = 0)
KLDmean2 = np.mean(KLDest2, axis = 0)
oKLDmean1 = np.mean(oKLDest1, axis = 0)
oKLDmean2 = np.mean(oKLDest2, axis = 0)

plot1 = plt.plot(epsset, KLDmean1, label = "Laplace (sampled)")
plot2 = plt.plot(epsset, KLDmean2, label = "Gaussian (sampled)")
plot3 = plt.plot(epsset, oKLDmean1, label = "Laplace (ordered)")
plot4 = plt.plot(epsset, oKLDmean2, label = "Gaussian (ordered)")

plt.title("Effect of epsilon on noisy estimate of KLD")
plt.xlabel("Value of epsilon")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()