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

for trial in range(4):

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial % 4 == 0 or trial % 4 == 1:
        p = dis.Laplace(loc = 0.1, scale = 1)

    # option 1b: distributions have large KL divergence
    else:
        p = dis.Laplace(loc = 1, scale = 1)

    q = dis.Normal(loc = 0, scale = 1)
    knownKLD = dis.kl_divergence(p, q)

    # parameters for the addition of Laplace and Gaussian noise
    EPS = 0.1
    DTA = 0.1
    A = 0
    R = 10

    # option 2a: baseline case
    if trial % 2 == 0:
        b1 = log(2) / EPS

    # option 2b: Monte-Carlo estimate
    else:
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
    sEstL = np.zeros((C, L))
    sEstN = np.zeros((C, L))
    oEstL = np.zeros((C, L))
    oEstN = np.zeros((C, L))
    print("Evaluating KL Divergence estimator...\n")

    with alive_bar(C*L) as bar:
        for T in Tset:

            sTotalNoiseL = 0
            sTotalNoiseN = 0
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
                    sTotalNoiseL = sTotalNoiseL + (noiseL.sample(sample_shape = (qT,)))
                    sTotalNoiseN = sTotalNoiseN + (noiseN.sample(sample_shape = (qT,)))
                    oTotalNoiseL = oTotalNoiseL + (noiseL.sample(sample_shape = (qOrderedT,)))
                    oTotalNoiseN = oTotalNoiseN + (noiseN.sample(sample_shape = (qOrderedT,)))

                sAvNoiseL = sTotalNoiseL / R
                sAvNoiseN = sTotalNoiseN / R
                oAvNoiseL = oTotalNoiseL / R
                oAvNoiseN = oTotalNoiseN / R

                # option 3a: add average noise term to unknown distribution
                sLogrL = p.log_prob(qClientSamp + sAvNoiseL) - q.log_prob(qClientSamp)
                sLogrN = p.log_prob(qClientSamp + sAvNoiseN) - q.log_prob(qClientSamp)
                oLogrL = p.log_prob(qOrdClientSamp + oAvNoiseL) - q.log_prob(qOrdClientSamp)
                oLogrN = p.log_prob(qOrdClientSamp + oAvNoiseN) - q.log_prob(qOrdClientSamp)

                # compute k3 estimator
                sK3noiseL = (sLogrL.exp() - 1) - sLogrL
                sK3noiseN = (sLogrN.exp() - 1) - sLogrN
                oK3noiseL = (oLogrL.exp() - 1) - oLogrL
                oK3noiseN = (oLogrN.exp() - 1) - oLogrN

                # compare with known KL divergence
                sEstL[j, T_COUNT] = abs(sK3noiseL.mean() - knownKLD)
                sEstN[j, T_COUNT] = abs(sK3noiseN.mean() - knownKLD)
                oEstL[j, T_COUNT] = abs(oK3noiseL.mean() - knownKLD)
                oEstN[j, T_COUNT] = abs(oK3noiseN.mean() - knownKLD)
            
                if T_COUNT < L - 1:
                    T_COUNT = T_COUNT + 1
                else:
                    T_COUNT = 0

                bar()

    # compute mean of unbiased estimator for particular T across all clients
    if trial % 4 == 0:
        sMeanL1 = np.mean(sEstL, axis = 0)
        sMeanN1 = np.mean(sEstN, axis = 0)
        oMeanL1 = np.mean(oEstL, axis = 0)
        oMeanN1 = np.mean(oEstN, axis = 0)

    if trial % 4 == 1:
        sMeanL2 = np.mean(sEstL, axis = 0)
        sMeanN2 = np.mean(sEstN, axis = 0)
        oMeanL2 = np.mean(oEstL, axis = 0)
        oMeanN2 = np.mean(oEstN, axis = 0)

    if trial % 4 == 2:
        sMeanL3 = np.mean(sEstL, axis = 0)
        sMeanN3 = np.mean(sEstN, axis = 0)
        oMeanL3 = np.mean(oEstL, axis = 0)
        oMeanN3 = np.mean(oEstN, axis = 0)

    if trial % 4 == 3:
        sMeanL4 = np.mean(sEstL, axis = 0)
        sMeanN4 = np.mean(sEstN, axis = 0)
        oMeanL4 = np.mean(oEstL, axis = 0)
        oMeanN4 = np.mean(oEstN, axis = 0)

# separate graphs for Laplace / Gaussian and Small / Large KL divergence to show trends
fig = plt.figure(figsize = (12.8, 9.6))

ax1 = plt.subplot(121)
ax1.plot(Tset, sMeanL1, label = "Small KLD + Lap (samp)")
ax1.plot(Tset, oMeanL1, label = "Small KLD + Lap (ord)")
ax1.plot(Tset, sMeanL2, label = "Small KLD + Lap (samp) mc")
ax1.plot(Tset, oMeanL2, label = "Small KLD + Lap (ord) mc")
ax1.set_title("Effect of T on error of unbiased estimator")
ax1.set_xlabel("Value of T")
ax1.set_ylabel("Error of unbiased estimator (mid noise)")
ax1.set_yscale("log")
ax1.legend(loc = "best")

ax2 = plt.subplot(122)
ax2.plot(Tset, sMeanN1, label = "Small KLD + Gauss (samp)")
ax2.plot(Tset, oMeanN1, label = "Small KLD + Gauss (ord)")
ax2.plot(Tset, sMeanN2, label = "Small KLD + Gauss (samp) mc")
ax2.plot(Tset, oMeanN2, label = "Small KLD + Gauss (ord) mc")
ax2.set_title("Effect of T on error of unbiased estimator")
ax2.set_xlabel("Value of T")
ax2.set_ylabel("Error of unbiased estimator (mid noise)")
ax2.set_yscale("log")
ax2.legend(loc = "best")

ax3 = plt.subplot(221)
ax3.plot(Tset, sMeanL3, label = "Large KLD + Lap (samp)")
ax3.plot(Tset, oMeanL3, label = "Large KLD + Lap (ord)")
ax3.plot(Tset, sMeanL4, label = "Large KLD + Lap (samp) mc")
ax3.plot(Tset, oMeanL4, label = "Large KLD + Lap (ord) mc")
ax3.set_title("Effect of T on error of unbiased estimator")
ax3.set_xlabel("Value of T")
ax3.set_ylabel("Error of unbiased estimator (mid noise)")
ax3.set_yscale("log")
ax3.legend(loc = "best")

ax4 = plt.subplot(222)
ax4.plot(Tset, sMeanN3, label = "Large KLD + Gauss (samp)")
ax4.plot(Tset, oMeanN3, label = "Large KLD + Gauss (ord)")
ax4.plot(Tset, sMeanN4, label = "Large KLD + Gauss (samp) mc")
ax4.plot(Tset, oMeanN4, label = "Large KLD + Gauss (ord) mc")
ax4.set_title("Effect of T on error of unbiased estimator")
ax4.set_xlabel("Value of T")
ax4.set_ylabel("Error of unbiased estimator (mid noise)")
ax4.set_yscale("log")
ax4.legend(loc = "best")

plt.tight_layout()
plt.savefig("plot_mid_noise_t.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
