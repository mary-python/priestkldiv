"""This package contains parameterisable probability distributions and sampling functions."""
import torch.distributions as dis
p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)
x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)
print("true", truekl)
logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr
k4 = ((logr.exp())*logr) - (logr.exp() - 1)
for k in (k1, k2, k3, k4):
    print((k.mean() - truekl) / truekl, k.std() / truekl)
