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

# order the pre-processed sample
qOrderedRound = torch.sort(qRound)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# use multi-dimensional numpy arrays to save sampled points and statistics
qClientSample = np.zeros((C, N))
qOrderedClientSample = np.zeros((C, N))
qT = np.zeros(C)
qOrderedT = np.zeros(C)
logr = np.zeros(C)
Ologr = np.zeros(C)
k3 = np.zeros(C)
Ok3 = np.zeros(C)

# parameters for the addition of Laplace and Gaussian noise
epsset = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4])
dta = 0.1
a = 0
b1 = log(2)
b2 = 2*((log(1.25))/dta)*b1

# noise distributions and multi-dimensional numpy arrays to store KLD estimates
noise1 = tfp.distributions.Laplace(loc=a, scale=b1)
noise2 = tfp.distributions.Normal(loc=a, scale=b2)
L = np.size(epsset)
KLDest1 = np.zeros((C, L))
KLDest2 = np.zeros((C, L))
OKLDest1 = np.zeros((C, L))
OKLDest2 = np.zeros((C, L))

for j in range(0, C):

    # sample from the pre-processed sample
    qClientSample[j] = qRound.sample(sample_shape=(N,))
    qT[j] = torch.numel(qClientSample[j])

    # each client gets 500 points in order from ordered pre-processed sample
    qOrderedClientSample[j] = qOrderedRound[(500*j):(500*(j+1))]
    qOrderedT[j] = torch.numel(qOrderedClientSample[j])

    # skip Prio step for now
    logr[j] = (p.log_prob(qClientSample[j]) - q.log_prob(qClientSample[j]))
    Ologr[j] = (p.log_prob(qOrderedClientSample[j]) - q.log_prob(qOrderedClientSample[j]))
    Ok3[j] = ((Ologr[j].exp() - 1) - Ologr[j])
    epsCount = 0

    for eps in epsset:

        # create separate numpy arrays for sampled and ordered cases
        k3noise1 = np.zeros(L)
        k3noise2 = np.zeros(L)
        Ok3noise1 = np.zeros(L)
        Ok3noise2 = np.zeros(L)

        # add average of 10 possible noise terms
        for k in range(0, 10):
            k3noise1[k] = k3 + (noise1.sample(sample_shape=qT[j], seed=12))/eps
            k3noise2[k] = k3 + (noise2.sample(sample_shape=qT[j], seed=12))/eps
            Ok3noise1[k] = Ok3 + (noise1.sample(sample_shape=qOrderedT[j], seed=12))/eps
            Ok3noise2[k] = Ok3 + (noise2.sample(sample_shape=qOrderedT[j], seed=12))/eps

        average1 = np.sum(k3noise1) / np.size(k3noise1)
        average2 = np.sum(k3noise2) / np.size(k3noise2)
        Oaverage1 = np.sum(Ok3noise1) / np.size(Ok3noise1)
        Oaverage2 = np.sum(Ok3noise2) / np.size(Ok3noise2)

        # compare with true KLD
        KLDest1[C, epsCount] = abs(np.mean(average1) - truekl)
        KLDest2[C, epsCount] = abs(np.mean(average2) - truekl)
        OKLDest1[C, epsCount] = abs(np.mean(Oaverage1) - truekl)
        OKLDest2[C, epsCount] = abs(np.mean(Oaverage2) - truekl)
        epsCount = epsCount + 1

# compute mean of KLD for particular epsilon across all clients
KLDmean1 = np.mean(KLDest1, axis = 0)
KLDmean2 = np.mean(KLDest2, axis = 0)
OKLDmean1 = np.mean(OKLDest1, axis = 0)
OKLDmean2 = np.mean(OKLDest2, axis = 0)

plot1 = plt.plot(epsset, KLDmean1, label = f"Laplace (sampled)")
plot2 = plt.plot(epsset, KLDmean2, label = f"Gaussian (sampled)")
plot3 = plt.plot(epsset, OKLDmean1, label = f"Lapplace (ordered)")
plot4 = plt.plot(epsset, OKLDmean2, label = f"Gaussian (ordered)")

plt.title("Effect of epsilon on noisy estimate of KLD")
plt.xlabel("Value of epsilon")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()