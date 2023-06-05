import torch
import torch.distributions as dis
from math import log
import tensorflow_probability as tfp
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = dis.Laplace(loc=0.1, scale=1)
q = dis.Normal(loc=0, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# sample T points
T = 500_000

# round to 2 d.p., find indices of and eliminate unique values
pSample = p.sample(sample_shape=(T,))
pRound = torch.round(pSample, decimals=2)
pUnique = torch.unique(pRound, return_counts=True)
pIndices = (pUnique[1] == 1).nonzero().flatten()
pUniqueIndices = pUnique[0][pIndices]

for i in pUniqueIndices:
    pRound = pRound[pRound != i]

pT = torch.numel(pRound)

# skip Prio step for now
logr = (p.log_prob(pRound) - q.log_prob(pRound))
k3 = ((logr.exp() - 1) - logr)
print(f'Approx vs true KLD (no noise): {(k3.mean() - truekl) / truekl, k3.std() / truekl}')

# add Laplace or Gaussian noise with parameter(s) eps (and dta)
eps = 0.1
dta = 0.1
a = 0
b1 = log(2)
b2 = 2*((log(1.25))/dta)*b1
noise = tfp.distributions.Laplace(loc=a, scale=b1)
# noise = tfp.distributions.Normal(loc=a, scale=b2)
k3noise = k3 + (noise.sample(sample_shape=pT))/eps
print(f'Approx KLD (noise vs no noise): {(np.mean(k3noise) - k3.mean()) / k3.mean(), np.std(k3noise) / k3.std()}')

# additional comparisons with true KLD
trueklnoise = truekl + noise.sample(sample_shape=pT)
print(f'Approx vs true KLD (with noise): {(np.mean(k3noise) - np.mean(trueklnoise)) / np.mean(trueklnoise), np.std(k3noise) / np.std(trueklnoise)}')
print(f'True KLD (noise vs no noise): {(np.mean(trueklnoise) - truekl) / truekl, np.std(trueklnoise) / truekl}')