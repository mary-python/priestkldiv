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

    # numpy arrays
    F = 5_000
    Tset = np.array([1*F, 2*F, 5*F, 10*F, 20*F, 50*F, 100*F, 200*F, 500*F, 1_000*F])
    L = np.size(Tset)
    sEst = np.zeros(C)
    oEst = np.zeros(C)
    print("Evaluating KL Divergence estimator...\n")

    with alive_bar(C*L) as bar:
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

                # each client gets 500 points in order from ordered pre-processed sample
                # translated by 1 every time and added mod 4419 to stay below upper bound 4918
                qOrdClientSamp = qOrderedRound[0][(j % 4419) : (j % 4419) + 500]

                # compute ratio between unknown and known distributions
                sLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
                oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)

                # compute k3 estimator
                sK3noise = (sLogr.exp() - 1) - sLogr
                oK3noise = (oLogr.exp() - 1) - oLogr

                # compare with known KL divergence
                sEst[j] = abs(sK3noise.mean() - knownKLD)
                oEst[j] = abs(oK3noise.mean() - knownKLD)
                bar()

    # compute mean of unbiased estimator across all clients
    sMean = np.mean(sEst)
    oMean = np.mean(oEst)

    # numpy arrays
    sMeanL = np.zeros(L)
    sMeanN = np.zeros(L)
    oMeanL = np.zeros(L)
    oMeanN = np.zeros(L)
    T_COUNT = 0

    for T in Tset:

        sTotalNoiseL = 0
        sTotalNoiseN = 0
        oTotalNoiseL = 0
        oTotalNoiseN = 0

        # compute average of R possible noise terms
        for k in range(0, R):
            sTotalNoiseL = sTotalNoiseL + (noiseL.sample(sample_shape = (1,)))
            sTotalNoiseN = sTotalNoiseN + (noiseN.sample(sample_shape = (1,)))
            oTotalNoiseL = oTotalNoiseL + (noiseL.sample(sample_shape = (1,)))
            oTotalNoiseN = oTotalNoiseN + (noiseN.sample(sample_shape = (1,)))

        sAvNoiseL = sTotalNoiseL / R
        sAvNoiseN = sTotalNoiseN / R
        oAvNoiseL = oTotalNoiseL / R
        oAvNoiseN = oTotalNoiseN / R

        # option 3b: add average noise term to final result
        sMeanL[T_COUNT] = sMean + sAvNoiseL
        sMeanN[T_COUNT] = sMean + sAvNoiseN
        oMeanL[T_COUNT] = sMean + oAvNoiseL
        oMeanN[T_COUNT] = sMean + oAvNoiseN
        T_COUNT = T_COUNT + 1

if trial % 4 == 0:
    sMeanL1 = sMeanL
    sMeanN1 = sMeanN
    oMeanL1 = oMeanL
    oMeanN1 = oMeanL

if trial % 4 == 1:
    sMeanL2 = sMeanL
    sMeanN2 = sMeanN
    oMeanL2 = oMeanL
    oMeanN2 = oMeanL

if trial % 4 == 2:
    sMeanL3 = sMeanL
    sMeanN3 = sMeanN
    oMeanL3 = oMeanL
    oMeanN3 = oMeanL

if trial % 4 == 3:
    sMeanL4 = sMeanL
    sMeanN4 = sMeanN
    oMeanL4 = oMeanL
    oMeanN4 = oMeanL

# separate graphs for Small / Large KL divergence to show trends
fig = plt.figure(figsize = (12.8, 4.8))

ax1 = plt.subplot(121)
ax1.plot(Tset, sMeanL1, label = "Small KLD + Lap (samp)")
ax1.plot(Tset, sMeanN1, label = "Small KLD + Gauss (samp)")
ax1.plot(Tset, oMeanL1, label = "Small KLD + Lap (ord)")
ax1.plot(Tset, oMeanN1, label = "Small KLD + Gauss (ord)")
ax1.plot(Tset, sMeanL2, label = "Small KLD + Lap (samp) mc")
ax1.plot(Tset, sMeanN2, label = "Small KLD + Gauss (samp) mc")
ax1.plot(Tset, oMeanL2, label = "Small KLD + Lap (ord) mc")
ax1.plot(Tset, oMeanN2, label = "Small KLD + Gauss (ord) mc")

ax1.set_title("Effect of T on error of unbiased estimator")
ax1.set_xlabel("Value of T")
ax1.set_ylabel("Error of unbiased estimator (end noise)")
ax1.set_yscale("log")
ax1.legend(loc = "upper center")

ax2 = plt.subplot(122)
ax2.plot(Tset, sMeanL3, label = "Large KLD + Lap (samp)")
ax2.plot(Tset, sMeanN3, label = "Large KLD + Gauss (samp)")
ax2.plot(Tset, oMeanL3, label = "Large KLD + Lap (ord)")
ax2.plot(Tset, oMeanN3, label = "Large KLD + Gauss (ord)")
ax2.plot(Tset, sMeanL4, label = "Large KLD + Lap (samp) mc")
ax2.plot(Tset, sMeanN4, label = "Large KLD + Gauss (samp) mc")
ax2.plot(Tset, oMeanL4, label = "Large KLD + Lap (ord) mc")
ax2.plot(Tset, oMeanN4, label = "Large KLD + Gauss (ord) mc")

ax2.set_title("Effect of T on error of unbiased estimator")
ax2.set_xlabel("Value of T")
ax2.set_ylabel("Error of unbiased estimator (end noise)")
ax2.set_yscale("log")
ax2.legend(loc = "upper center")

plt.tight_layout()
plt.savefig("plot_end_noise_t.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
