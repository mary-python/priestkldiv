"""To compute the natural logarithm of a number, parameterise probability distributions,
create static, animated, and interactive visualisations, and work with arrays."""
from math import log
import torch
import torch.distributions as dis
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(12)

# p is private distribution, q is public
# option 1b: KLD is large
p = dis.Laplace(loc = 1, scale = 1)
q = dis.Normal(loc = 0, scale = 1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# parameters for the addition of Laplace and Gaussian noise
EPS = 0.1
DTA = 0.1
A = 0
R = 10

# option 2a: querying entire distribution
b1 = log(2) / EPS
b2 = (2*((log(1.25))/DTA)*b1) / EPS

# load Laplace and Normal noise distributions, dependent on eps
noiseL = dis.Laplace(loc = A, scale = b1)
noiseN = dis.Normal(loc = A, scale = b2)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# numpy arrays
F = 5_000
Tset = np.array([1*F, 2*F, 5*F, 10*F, 20*F, 50*F, 100*F, 200*F, 500*F, 1_000*F])
L = np.size(Tset)
KLDest = np.zeros(C)
oKLDest = np.zeros(C)

for T in Tset:

    # round to 2 d.p., find indices of and eliminate unique values
    qSample = q.sample(sample_shape = (T,))
    qRound = torch.round(qSample, decimals = 2)
    qUnique = torch.unique(qRound, return_counts = True)
    qIndices = (qUnique[1] == 1).nonzero().flatten()
    qUniqueIndices = qUnique[0][qIndices]

    for i in qUniqueIndices:
        qRound = qRound[qRound != i]

    # order the pre-processed sample and separate into + and - values
    qOrderedRound = torch.sort(qRound)
    qNegativeRound = qOrderedRound[0][0:2440]
    qPositiveRound = qOrderedRound[0][2440:4918]

    for j in range(0, C):

        # even clients get positive values, odd clients get negative values
        if (j % 2) == 0:
            indices = torch.randperm(len(qPositiveRound))[:N]
            qClientSamp = qPositiveRound[indices]
        else:
            indices = torch.randperm(len(qNegativeRound))[:N]
            qClientSamp = qNegativeRound[indices]

        qT1 = torch.numel(qClientSamp)
        qClientSamp = qClientSamp[abs(int(qT1 - 0.98*T)):]
        qT2 = torch.numel(qClientSamp)

        # each client gets 500 points in order from ordered pre-processed sample
        qOrdClientSamp = qOrderedRound[0][500*j : 500*(j + 1)]
        qOrderedT1 = torch.numel(qOrdClientSamp)
        qOrdClientSamp = qOrdClientSamp[abs(int(qOrderedT1 - 0.98*T)):]
        qOrderedT2 = torch.numel(qOrdClientSamp)

        # compute ratio between private and public distributions
        logr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
        oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)

        # compute k3 estimator
        k3noise = (logr.exp() - 1) - logr
        oK3noise = (oLogr.exp() - 1) - oLogr

        # compare with true KLD
        KLDest[j] = abs(k3noise.mean() - truekl)
        oKLDest[j] = abs(oK3noise.mean() - truekl)

# compute mean of KLD for particular T across all clients
KLDmean = np.mean(KLDest, axis = 0)
oKLDmean = np.mean(oKLDest, axis = 0)

# numpy arrays to store noisy KLD means
KLDmeanL = np.zeros(L)
KLDmeanN = np.zeros(L)
oKLDmeanL = np.zeros(L)
oKLDmeanN = np.zeros(L)
T_COUNT = 0

for T in Tset:

    totalNoiseL = 0
    totalNoiseN = 0
    oTotalNoiseL = 0
    oTotalNoiseN = 0

    # compute average of R possible noise terms
    for k in range(0, R):
        totalNoiseL = totalNoiseL + (noiseL.sample(sample_shape = (qT2*C,)))
        totalNoiseN = totalNoiseN + (noiseN.sample(sample_shape = (qT2*C,)))
        oTotalNoiseL = oTotalNoiseL + (noiseL.sample(sample_shape = (qOrderedT2*C,)))
        oTotalNoiseN = oTotalNoiseN + (noiseN.sample(sample_shape = (qOrderedT2*C,)))

    avNoiseL = totalNoiseL / R
    avNoiseN = totalNoiseN / R
    oAvNoiseL = oTotalNoiseL / R
    oAvNoiseN = oTotalNoiseN / R

    # option 3b: add average noise term to final result
    KLDmeanL[T_COUNT] = KLDmean[T_COUNT] + avNoiseL
    KLDmeanN[T_COUNT] = KLDmean[T_COUNT] + avNoiseN
    oKLDmeanL[T_COUNT] = oKLDmean[T_COUNT] + oAvNoiseL
    oKLDmeanN[T_COUNT] = oKLDmean[T_COUNT] + oAvNoiseN
    T_COUNT = T_COUNT + 1

plot1 = plt.plot(Tset, KLDmeanL, label = "Laplace (sampled)")
plot2 = plt.plot(Tset, KLDmeanN, label = "Gaussian (sampled)")
plot3 = plt.plot(Tset, oKLDmeanL, label = "Laplace (ordered)")
plot4 = plt.plot(Tset, oKLDmeanN, label = "Gaussian (ordered)")

plt.title("Effect of T on noisy estimate of KLD")
plt.xlabel("Value of T")
plt.ylabel("Difference in KLD")
plt.yscale('log')
plt.legend(loc = "best")
plt.show()
