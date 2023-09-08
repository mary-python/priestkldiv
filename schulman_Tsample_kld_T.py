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

eps = 0.1
dta = 0.1
a = 0
b1 = log(2)
b2 = 2*((log(1.25))/dta)*b1

noise1 = tfp.distributions.Laplace(loc=a, scale=b1)
noise2 = tfp.distributions.Normal(loc=a, scale=b2)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# use multi-dimensional numpy arrays to save sampled points and statistics
qClientSample = np.zeros((N, C))
qT1 = np.zeros(C)
qT2 = np.zeros(C)
logr = np.zeros(C)
k3 = np.zeros(C)

# store T before C (clients)
Tset = np.array([5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000])
L = np.size(Tset)
KLDest1 = np.zeros((L, C))
KLDest2 = np.zeros((L, C))
TCount = 0

for T in Tset:

    k3noise1 = np.zeros(L)
    k3noise2 = np.zeros(L)

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

        for j in range(0, C):

            # sample from the pre-processed sample
            qClientSample[j] = qSample.sample(sample_shape=(N,))
            qT1[j] = torch.numel(qClientSample[j])
            qClientSample[j] = qClientSample[j][abs(int(qT1 - 0.98*T)):]
            qT2[j] = torch.numel(qClientSample[j])
    
            # skip Prio step for now
            logr[j] = (p.log_prob(qClientSample[j]) - q.log_prob(qClientSample[j]))
            k3[j] = ((logr[j].exp() - 1) - logr[j])
        
            # add Laplace and Gaussian noise with parameter(s) eps (and dta)
            k3noise1[k] = k3 + (noise1.sample(sample_shape=qT2[j], seed=12)/eps)
            k3noise2[k] = k3 + (noise2.sample(sample_shape=qT2[j], seed=12)/eps)

        average1 = np.sum(k3noise1) / np.size(k3noise1)
        average2 = np.sum(k3noise2) / np.size(k3noise2)

        # compare with true KLD
        KLDest1[T, C] = abs(np.mean(average1) - truekl)
        KLDest2[T, C] = abs(np.mean(average2) - truekl)
        TCount = TCount + 1

# compute mean of KLD for particular T across all clients
KLDmean1 = np.mean(KLDest1, axis = 1)
KLDmean2 = np.mean(KLDest2, axis = 1)

plot1 = plt.plot(Tset, KLDest1, label = f"Laplace dist")
plot2 = plt.plot(Tset, KLDest2, label = f"Gaussian dist")

plt.title("Effect of T on noisy estimate of KLD")
plt.xlabel("Value of T")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()