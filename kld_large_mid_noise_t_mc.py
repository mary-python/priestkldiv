"""Modules provide various time-related functions as well as progress updates,
compute the natural logarithm of a number, parameterise probability distributions,
create static, animated, and interactive visualisations, and work with arrays."""
import time
from math import log
from alive_progress import alive_bar
import torch
import torch.distributions as dis
import matplotlib.pyplot as plt
import numpy as np

# initialising start time and seed for random sampling
startTime = time.perf_counter()
print("\nStarting...")
torch.manual_seed(12)

# p is private distribution, q is public
# option 1b: KLD is large
p = dis.Laplace(loc = 1, scale = 1)
q = dis.Normal(loc = 0, scale = 1)
truekl = dis.kl_divergence(p, q)

# parameters for the addition of Laplace and Gaussian noise
EPS = 0.1
DTA = 0.1
A = 0
R = 10

# option 2b: Monte-Carlo sampling
b1 = (1 + log(2)) / EPS
b2 = (2*((log(1.25))/DTA)*b1) / EPS

# load Laplace and Normal noise distributions, dependent on eps
noiseL = dis.Laplace(loc = A, scale = b1)
noiseN = dis.Normal(loc = A, scale = b2)

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# multi-dimensional numpy arrays
F = 5_000
Tset = np.array([1*F, 2*F, 5*F, 10*F, 20*F, 50*F, 100*F, 200*F, 500*F, 1_000*F])
L = np.size(Tset)
KLDestL = np.zeros((C, L))
KLDestN = np.zeros((C, L))
oKLDestL = np.zeros((C, L))
oKLDestN = np.zeros((C, L))
print("Evaluating KL Divergence estimator...\n")

with alive_bar(C*L) as bar:
    for T in Tset:

        totalNoiseL = 0
        totalNoiseN = 0
        oTotalNoiseL = 0
        oTotalNoiseN = 0

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
        T_COUNT = 0

        for j in range(0, C):

            # even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                indices = torch.randperm(len(qPositiveRound))[:N]
                qClientSamp = qPositiveRound[indices]
            else:
                indices = torch.randperm(len(qNegativeRound))[:N]
                qClientSamp = qNegativeRound[indices]

            qT = torch.numel(qClientSamp)

            # each client gets 500 points in order from ordered pre-processed sample
            # translated by 1 every time and added mod 4419 to stay below upper bound 4918
            qOrdClientSamp = qOrderedRound[0][(j % 4419) : (j % 4419) + 500]
            qOrderedT = torch.numel(qOrdClientSamp)

            # compute average of R possible noise terms
            for k in range(0, R):
                totalNoiseL = totalNoiseL + (noiseL.sample(sample_shape = (qT,)))
                totalNoiseN = totalNoiseN + (noiseN.sample(sample_shape = (qT,)))
                oTotalNoiseL = oTotalNoiseL + (noiseL.sample(sample_shape = (qOrderedT,)))
                oTotalNoiseN = oTotalNoiseN + (noiseN.sample(sample_shape = (qOrderedT,)))

            avNoiseL = totalNoiseL / R
            avNoiseN = totalNoiseN / R
            oAvNoiseL = oTotalNoiseL / R
            oAvNoiseN = oTotalNoiseN / R

            # option 3a: add average noise term to private distribution
            logrL = p.log_prob(qClientSamp + avNoiseL) - q.log_prob(qClientSamp)
            logrN = p.log_prob(qClientSamp + avNoiseN) - q.log_prob(qClientSamp)
            oLogrL = p.log_prob(qOrdClientSamp + oAvNoiseL) - q.log_prob(qOrdClientSamp)
            oLogrN = p.log_prob(qOrdClientSamp + oAvNoiseN) - q.log_prob(qOrdClientSamp)

            # compute k3 estimator
            k3noiseL = (logrL.exp() - 1) - logrL
            k3noiseN = (logrN.exp() - 1) - logrN
            oK3noiseL = (oLogrL.exp() - 1) - oLogrL
            oK3noiseN = (oLogrN.exp() - 1) - oLogrN

            # compare with true KLD
            KLDestL[j, T_COUNT] = abs(k3noiseL.mean() - truekl)
            KLDestN[j, T_COUNT] = abs(k3noiseN.mean() - truekl)
            oKLDestL[j, T_COUNT] = abs(oK3noiseL.mean() - truekl)
            oKLDestN[j, T_COUNT] = abs(oK3noiseN.mean() - truekl)
            
            if T_COUNT < L - 1:
                T_COUNT = T_COUNT + 1
            else:
                T_COUNT = 0

            bar()

# compute mean of KLD for particular T across all clients
KLDmeanL = np.mean(KLDestL, axis = 0)
KLDmeanN = np.mean(KLDestN, axis = 0)
oKLDmeanL = np.mean(oKLDestL, axis = 0)
oKLDmeanN = np.mean(oKLDestN, axis = 0)

# separate graphs for Laplace / Gaussian to show trends
fig = plt.figure(figsize = (12.8, 4.8))

ax1 = plt.subplot(121)
ax1.plot(Tset, KLDmeanL, label = "Laplace (sampled)")
ax1.plot(Tset, oKLDmeanL, label = "Laplace (ordered)")
ax1.set_title("Effect of T on noisy estimate of KLD")
ax1.set_xlabel("Value of T")
ax1.set_ylabel("Difference in KLD")
ax1.set_yscale("log")
ax1.legend(loc = "best")

ax2 = plt.subplot(122)
ax2.plot(Tset, KLDmeanN, label = "Gaussian (sampled)")
ax2.plot(Tset, oKLDmeanN, label = "Gaussian (ordered)")
ax2.set_title("Effect of T on noisy estimate of KLD")
ax2.set_xlabel("Value of T")
ax2.set_ylabel("Difference in KLD")
ax2.set_yscale("log")
ax2.legend(loc = "best")

plt.tight_layout()
plt.ion()
plt.show()
plt.pause(0.001)
input("\nPress [enter] to continue.")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
