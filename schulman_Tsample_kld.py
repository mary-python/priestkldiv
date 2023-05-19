import torch
import torch.distributions as dis
from math import log
import tensorflow_probability as tfp
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=1, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# sample T = 10 million points
x = q.sample(sample_shape=(10_000_000,))
y = x.clone().detach()
z = torch.cat((x, y))

# skip Prio step for now
logr = (p.log_prob(z) - q.log_prob(z))
k3 = ((logr.exp() - 1) - logr)
print(f'Approx vs true KLD (no noise): {(k3.mean() - truekl) / truekl, k3.std() / truekl}')

# add Laplace noise with eps = 1 (full distribution)
a, b1 = 0, log(2)
noise = tfp.distributions.Laplace(loc=a, scale=b1)
k3noise = k3 + noise.sample(sample_shape=20_000_000)
print(f'Approx KLD (noise vs no noise): {(np.mean(k3noise) - k3.mean()) / k3.mean(), np.std(k3noise) / k3.std()}')

trueklnoise = truekl + noise.sample(sample_shape=20_000_000)
print(f'Approx vs true KLD (with noise): {(np.mean(k3noise) - np.mean(trueklnoise)) / np.mean(trueklnoise), np.std(k3noise) / np.std(trueklnoise)}')
print(f'True KLD (noise vs no noise): {(np.mean(trueklnoise) - truekl) / truekl, np.std(trueklnoise) / truekl}')