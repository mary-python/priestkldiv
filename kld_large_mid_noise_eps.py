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

# sample T points
T = 500_000

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
qNegativeRound = qOrderedRound[0][0:249151]
qPositiveRound = qOrderedRound[0][249151:499947]

# 10-100K clients each with a few hundred points
C = 10000
N = 500

# parameters for the addition of Laplace and Gaussian noise
epsset = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4])
DTA = 0.1
A = 0
R = 10

# option 2a: querying entire distribution
b1 = log(2)
b2 = 2*((log(1.25))/DTA)*b1

# multi-dimensional numpy arrays to store KLD estimates
L = np.size(epsset)
KLDestL = np.zeros((C, L))
KLDestN = np.zeros((C, L))
oKLDestL = np.zeros((C, L))
oKLDestN = np.zeros((C, L))
print("Evaluating KL Divergence estimator...\n")

with alive_bar(C*L) as bar:
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
        # translated by 49 every time to stay below upper bound 499947 for client 10000
        qOrdClientSamp = qOrderedRound[0][49*j : (49*j) + 500]
        qOrderedT = torch.numel(qOrdClientSamp)
        EPS_COUNT = 0

        for eps in epsset:

            totalNoiseL = 0
            totalNoiseN = 0
            oTotalNoiseL = 0
            oTotalNoiseN = 0

            # load Laplace and Normal noise distributions, dependent on eps
            s1 = b1 / eps
            s2 = b2 / eps
            noiseL = dis.Laplace(loc = A, scale = s1)
            noiseN = dis.Normal(loc = A, scale = s2)

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
            KLDestL[j, EPS_COUNT] = abs(k3noiseL.mean() - truekl)
            KLDestN[j, EPS_COUNT] = abs(k3noiseN.mean() - truekl)
            oKLDestL[j, EPS_COUNT] = abs(oK3noiseL.mean() - truekl)
            oKLDestN[j, EPS_COUNT] = abs(oK3noiseN.mean() - truekl)
            EPS_COUNT = EPS_COUNT + 1
            bar()

# compute mean of KLD for particular epsilon across all clients
KLDmeanL = np.mean(KLDestL, axis = 0)
KLDmeanN = np.mean(KLDestN, axis = 0)
oKLDmeanL = np.mean(oKLDestL, axis = 0)
oKLDmeanN = np.mean(oKLDestN, axis = 0)

plot1 = plt.plot(epsset, KLDmeanL, label = "Laplace (sampled)")
plot2 = plt.plot(epsset, KLDmeanN, label = "Gaussian (sampled)")
plot3 = plt.plot(epsset, oKLDmeanL, label = "Laplace (ordered)")
plot4 = plt.plot(epsset, oKLDmeanN, label = "Gaussian (ordered)")

plt.title("Effect of epsilon on noisy estimate of KLD")
plt.xlabel("Value of epsilon")
plt.ylabel("Difference in KLD")
plt.yscale("log")
plt.legend(loc = "best")

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
