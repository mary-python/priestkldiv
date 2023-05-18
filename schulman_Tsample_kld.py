import torch
import torch.distributions as dis
from math import log
import tensorflow_probability as tfp
import numpy as np

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# sample T = 10 million points
x = q.sample(sample_shape=(10_000_000,))
y = x.clone().detach()
z = torch.cat((x, y))

# skip Prio step for now
logr = (p.log_prob(z) - q.log_prob(z))
k3 = ((logr.exp() - 1) - logr)
print((k3.mean() - truekl) / truekl, k3.std() / truekl)

# add Laplace noise with eps = 1 (full distribution)
a, b1 = 0, log(2)
noise = tfp.distributions.Laplace(loc=a, scale=b1)
k3noise = k3 + noise.sample(sample_shape=20_000_000)
print((np.mean(k3noise) - truekl) / truekl, np.std(k3noise) / truekl)