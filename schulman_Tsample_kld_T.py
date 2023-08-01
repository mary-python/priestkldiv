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
KLDest1 = list()
KLDest2 = list()

# sample T points
Tset = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]

for T in Tset:

    k3noise1 = list()
    k3noise2 = list()

    for j in range(0, 10):

        # round to 2 d.p., find indices of and eliminate unique values
        qSample = q.sample(sample_shape=(T,))
        qRound = torch.round(qSample, decimals=2)
        qUnique = torch.unique(qRound, return_counts=True)
        qIndices = (qUnique[1] == 1).nonzero().flatten()
        qUniqueIndices = qUnique[0][qIndices]

        for i in qUniqueIndices:
            qRound = qRound[qRound != i]

        qT1 = torch.numel(qRound)
        qRound = qRound[abs(int(qT1 - 0.98*T)):]
        qT2 = torch.numel(qRound)
    
        # skip Prio step for now
        logr = (p.log_prob(qRound) - q.log_prob(qRound))
        k3 = ((logr.exp() - 1) - logr)
        
        # add Laplace and Gaussian noise with parameter(s) eps (and dta)
        k3noise1.append(k3 + (noise1.sample(sample_shape=qT2, seed=12)/eps))
        k3noise2.append(k3 + (noise2.sample(sample_shape=qT2, seed=12)/eps))

    average1 = sum(k3noise1) / len(k3noise1)
    average2 = sum(k3noise2) / len(k3noise2)
    KLDest1.append(abs(np.mean(average1) - truekl))
    KLDest2.append(abs(np.mean(average2) - truekl))

plot1 = plt.plot(Tset, KLDest1, label = f"Laplace dist")
plot2 = plt.plot(Tset, KLDest2, label = f"Gaussian dist")

plt.title("Effect of T on noisy estimate of KLD")
plt.xlabel("Value of T")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()