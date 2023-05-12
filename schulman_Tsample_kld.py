import torch
import torch.distributions as dis

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

x = q.sample(sample_shape=(10_000_000,))
y = x.clone().detach()
z = torch.cat((x, y))

logr = (p.log_prob(z) - q.log_prob(z))
k3 = ((logr.exp() - 1) - logr)
print((k3.mean() - truekl) / truekl, k3.std() / truekl)