import torch
torch.manual_seed(12)
import torch.distributions as dis
from math import log
import tensorflow as tf
tf.random.set_seed(638)
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = dis.Laplace(loc=0.1, scale=1)
q = dis.Normal(loc=0, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# parameters for the addition of Laplace and Gaussian noise
eps = 0.1
dta = 0.1
a = 0
b1 = log(2)
b2 = 2*((log(1.25))/dta)*b1

# noise distributions
noise1 = tfp.distributions.Laplace(loc=a, scale=b1)
noise2 = tfp.distributions.Normal(loc=a, scale=b2)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# use multi-dimensional numpy arrays to save sampled points and statistics
qClientSample = np.zeros((N, C))
qOrderedClientSample = np.zeros((N, C))
qT1 = np.zeros(C)
qT2 = np.zeros(C)
qOrderedT1 = np.zeros(C)
qOrderedT2 = np.zeros(C)
logr = np.zeros(C)
oLogr = np.zeros(C)
k3 = np.zeros(C)
oK3 = np.zeros(C)

# store T before C (clients)
Tset = np.array([5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000])
L = np.size(Tset)
KLDest1 = np.zeros((L, C))
KLDest2 = np.zeros((L, C))
oKLDest1 = np.zeros((L, C))
oKLDest2 = np.zeros((L, C))
TCount = 0

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
        qPositiveRound = torch.where(qRound >= 0, qRound, -qRound)
        qNegativeRound = torch.where(qRound < 0, qRound, -qRound)

        for j in range(0, C):

            # even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                qClientSample[j] = torch.multinomial(qPositiveRound, N, False)
            else:
                qClientSample[j] = -torch.multinomial(qNegativeRound, N, False)

            qT1[j] = torch.numel(torch.from_numpy(qClientSample[j]))
            qClientSample[j] = qClientSample[j][abs(int(qT1 - 0.98*T)):]
            qT2[j] = torch.numel(torch.from_numpy(qClientSample[j]))

            # each client gets 500 points in order from ordered pre-processed sample
            qOrderedClientSample[j] = qOrderedRound[0][(500*j):(500*(j+1))]
            qOrderedT1[j] = torch.numel(torch.from_numpy(qOrderedClientSample[j]))
            qOrderedClientSample[j] = qOrderedClientSample[j][abs(int(qOrderedT1 - 0.98*T)):]
            qT2[j] = torch.numel(torch.from_numpy(qOrderedClientSample[j]))
    
            # skip Prio step for now
            logr[j] = (p.log_prob(torch.from_numpy(qClientSample[j])) - q.log_prob(torch.from_numpy(qClientSample[j])))
            oLogr[j] = (p.log_prob(torch.from_numpy(qOrderedClientSample[j])) - q.log_prob(torch.from_numpy(qOrderedClientSample[j])))
            k3[j] = ((logr[j].exp() - 1) - logr[j])
            oK3[j] = ((oLogr[j].exp() - 1) - oLogr[j])
        
            # add Laplace and Gaussian noise with parameter(s) eps (and dta)
            k3noise1[k] = k3 + (noise1.sample(sample_shape=qT2[j], seed=12)/eps)
            k3noise2[k] = k3 + (noise2.sample(sample_shape=qT2[j], seed=12)/eps)
            oK3noise1[k] = oK3 + (noise1.sample(sample_shape=qOrderedT2[j], seed=12)/eps)
            oK3noise2[k] = oK3 + (noise2.sample(sample_shape=qOrderedT2[j], seed=12)/eps)

        average1 = np.sum(k3noise1) / np.size(k3noise1)
        average2 = np.sum(k3noise2) / np.size(k3noise2)
        oAverage1 = np.sum(oK3noise1) / np.size(oK3noise1)
        oAverage2 = np.sum(oK3noise2) / np.size(oK3noise2)

        # compare with true KLD
        KLDest1[TCount, C] = abs(np.mean(average1) - truekl)
        KLDest2[TCount, C] = abs(np.mean(average2) - truekl)
        oKLDest1[TCount, C] = abs(np.mean(oAverage1) - truekl)
        oKLDest2[TCount, C] = abs(np.mean(oAverage2) - truekl)
        TCount = TCount + 1

# compute mean of KLD for particular T across all clients
KLDmean1 = np.mean(KLDest1, axis = 1)
KLDmean2 = np.mean(KLDest2, axis = 1)
oKLDmean1 = np.mean(oKLDest1, axis = 1)
oKLDmean2 = np.mean(oKLDest2, axis = 1)

plot1 = plt.plot(Tset, KLDest1, label = f"Laplace (sampled)")
plot2 = plt.plot(Tset, KLDest2, label = f"Gaussian (sampled)")
plot3 = plt.plot(Tset, oKLDest1, label = f"Laplace (ordered)")
plot4 = plt.plot(Tset, oKLDest2, label = f"Gaussian (ordered)")

plt.title("Effect of T on noisy estimate of KLD")
plt.xlabel("Value of T")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()