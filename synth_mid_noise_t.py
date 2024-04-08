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

    # 200 clients each with 125 points
    C = 200
    N = 125

    # multi-dimensional numpy arrays
    F = 5_000
    Tset = np.array([1*F, 2*F, 5*F, 10*F, 20*F, 50*F, 100*F, 200*F, 500*F, 1_000*F])
    E = np.size(Tset)
    rLda = 1
    ldaStep = 0.05
    L = int(rLda / ldaStep)
    sEst = np.zeros((C, E, L))
    oEst = np.zeros((C, E, L))
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

                # each client gets N points in order from ordered pre-processed sample
                # translated by 5 every time to stay below upper bound 4918
                qOrdClientSamp = qOrderedRound[0][5*j : (5*j) + N]

                sLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
                oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)
                LDA_COUNT = 0

                # explore lambdas in a range
                for lda in range(0, rLda + ldaStep, ldaStep):

                    # compute k3 estimator
                    sRangeEst = (lda * (sLogr.exp() - 1)) - sLogr
                    oRangeEst = (lda * (oLogr.exp() - 1)) - oLogr

                    # share unbiased estimator with intermediate server
                    sEst[j, T_COUNT, LDA_COUNT] = sRangeEst.mean()
                    oEst[j, T_COUNT, LDA_COUNT] = oRangeEst.mean()
                    LDA_COUNT = LDA_COUNT + 1

                if T_COUNT < E - 1:
                    T_COUNT = T_COUNT + 1
                else:
                    T_COUNT = 0

                bar()

    sMeanLda = np.zeros((L, E))
    oMeanLda = np.zeros((L, E))

    # compute mean of unbiased estimator across clients
    for l in range(0, rLda + ldaStep, ldaStep):
        sMeanLda[l] = np.mean(sEst, axis = (0, 1))
        oMeanLda[l] = np.mean(oEst, axis = (0, 1))

    # compute mean of unbiased estimator across epsilon
    sOpt = np.mean(sMeanLda, axis = 1)
    oOpt = np.mean(sMeanLda, axis = 1)

    # find lambda that produces minimum error
    sIndex = np.argmin(sOpt)
    oIndex = np.argmin(oOpt)

    sLda = ldaStep * sIndex
    oLda = ldaStep * oIndex

    # mean across clients for optimum lambda
    sMean = sMeanLda[sLda]
    oMean = sMeanLda[oLda]

    if trial % 4 == 0:
        sMean1 = sMean
        oMean1 = oMean
    
    if trial % 4 == 1:
        sMean2 = sMean
        oMean2 = oMean

    if trial % 4 == 2:
        sMean3 = sMean
        oMean3 = oMean

    if trial % 4 == 3:
        sMean4 = sMean
        oMean4 = oMean

plt.plot(Tset, sMean1, label = "Small KLD (samp)")
plt.plot(Tset, oMean1, label = "Small KLD (ord)")
plt.plot(Tset, sMean2, label = "Small KLD (samp) mc")
plt.plot(Tset, oMean2, label = "Small KLD (ord) mc")
plt.plot(Tset, sMean3, label = "Large KLD (samp)")
plt.plot(Tset, oMean3, label = "Large KLD (ord)")
plt.plot(Tset, sMean4, label = "Large KLD (samp) mc")
plt.plot(Tset, oMean4, label = "Large KLD (ord) mc")

plt.title("Effect of T on error of unbiased estimator")
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (mid noise)")
plt.yscale("log")
plt.legend(loc = "best")
plt.savefig("plot_mid_noise_t.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
