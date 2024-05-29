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

for trial in range(12):
    print(f"\nTrial {trial + 1}...")

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial % 4 == 0 or trial % 4 == 1:
        p = dis.Laplace(loc = 0.1, scale = 1)

    # option 1b: distributions have large KL divergence
    else:
        p = dis.Laplace(loc = 1, scale = 1)

    q = dis.Normal(loc = 0, scale = 1)

    # error at end = (output of algorithm - ground truth) squared
    groundTruth = torch.distributions.kl.kl_divergence(p, q)

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

    # 200 clients each with 125 points
    C = 200
    N = 125

    # parameters for the addition of Laplace and Gaussian noise
    epsset = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4])
    DTA = 0.1
    A = 0
    R = 10
    
    # option 2a: baseline case
    if trial % 2 == 0:
        b1 = log(2)
    
    # option 2b: Monte-Carlo estimate
    else:
        b1 = 1 + log(2)

    b2 = 2*((log(1.25))/DTA)*b1

    # numpy arrays
    E = np.size(epsset)
    sMeanA = np.zeros(E)
    oMeanA = np.zeros(E)
    EPS_COUNT = 0

    for eps in epsset:
        print(f"Trial {trial + 1}: epsilon = {eps}...")

        # load Gaussian and Laplace noise distributions, dependent on eps
        s1 = b1 / eps
        s2 = b2 / eps

        if trial < 8:
            s3 = s2 * (np.sqrt(2) / R)
            probGaussNoise = dis.Normal(loc = A, scale = s3 / 100)
            gaussNoise = dis.Normal(loc = A, scale = s3)
        
        else:
            lapNoise = dis.Laplace(loc = A, scale = s1)

        # numpy arrays
        rLda = 1
        ldaStep = 0.05
        L = int((rLda + ldaStep) / ldaStep)
        sEst = np.zeros((L, C))
        oEst = np.zeros((L, C))

        for j in range(C):

            # even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                indices = torch.randperm(len(qPositiveRound))[:N]
                qClientSamp = qPositiveRound[indices]
            else:
                indices = torch.randperm(len(qNegativeRound))[:N]
                qClientSamp = qNegativeRound[indices]

            # each client gets N points in order from ordered pre-processed sample
            qOrdClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

            # option 3a: each client adds Gaussian noise term
            if trial < 4:
                sStartNoise = probGaussNoise.sample(sample_shape = (1,))
                oStartNoise = probGaussNoise.sample(sample_shape = (1,))

                if sStartNoise < 0:
                    qClientSamp = qClientSamp - sStartNoise
                else:
                    qClientSamp = qClientSamp + sStartNoise

                if oStartNoise < 0:
                    qClientSamp = qClientSamp - oStartNoise
                else:
                    qClientSamp = qClientSamp + oStartNoise

            # compute ratio between unknown and known distributions
            sLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
            oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)
            LDA_COUNT = 0

            # explore lambdas in a range
            for lda in np.arange(0, rLda + ldaStep, ldaStep):

                # compute k3 estimator
                sRangeEst = (lda * (np.exp(sLogr) - 1)) - sLogr
                oRangeEst = (lda * (np.exp(oLogr) - 1)) - oLogr

                # share unbiased estimator with server
                sEst[LDA_COUNT, j] = sRangeEst.mean()
                oEst[LDA_COUNT, j] = oRangeEst.mean()
                LDA_COUNT = LDA_COUNT + 1

        # compute mean of unbiased estimator across clients
        sMeanLda = np.mean(sEst, axis = 1)
        oMeanLda = np.mean(oEst, axis = 1)
        
        # option 3a: intermediate server adds Gaussian noise term
        if 4 <= trial < 8:
            for l in range(L):
                sMeanLda[l] = sMeanLda[l] + gaussNoise.sample(sample_shape = (1,))
                oMeanLda[l] = oMeanLda[l] + gaussNoise.sample(sample_shape = (1,))

        # find lambda that produces minimum error
        sIndex = np.argmin(sMeanLda)
        oIndex = np.argmin(oMeanLda)

        # mean across clients for optimum lambda
        sMean = sMeanLda[sIndex]
        oMean = oMeanLda[oIndex]

        # option 3b: server adds Laplace noise term to final result
        if trial >= 8:
            sMeanA[EPS_COUNT] = (sMean + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2
            oMeanA[EPS_COUNT] = (oMean + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2

        # option 3a: intermediate server has already added Gaussian noise term
        else:
            sMeanA[EPS_COUNT] = (sMean - groundTruth)**2
            oMeanA[EPS_COUNT] = (oMean - groundTruth)**2

        EPS_COUNT = EPS_COUNT + 1

    if trial % 12 == 0:
        sMean1 = sMeanA
        oMean1 = oMeanA

    if trial % 12 == 1:
        sMean2 = sMeanA
        oMean2 = oMeanA

    if trial % 12 == 2:
        sMean3 = sMeanA
        oMean3 = oMeanA

    if trial % 12 == 3:
        sMean4 = sMeanA
        oMean4 = oMeanA

    if trial % 12 == 4:
        sMean5 = sMeanA
        oMean5 = oMeanA
    
    if trial % 12 == 5:
        sMean6 = sMeanA
        oMean6 = oMeanA
    
    if trial % 12 == 6:
        sMean7 = sMeanA
        oMean7 = oMeanA

    if trial % 12 == 7:
        sMean8 = sMeanA
        oMean8 = oMeanA

    if trial % 12 == 8:
        sMean9 = sMeanA
        oMean9 = oMeanA
    
    if trial % 12 == 9:
        sMean10 = sMeanA
        oMean10 = oMeanA
    
    if trial % 12 == 10:
        sMean11 = sMeanA
        oMean11 = oMeanA

    if trial % 12 == 11:
        sMean12 = sMeanA
        oMean12 = oMeanA

# separate graphs for Small / Large KL divergence and start / mid / end noise to show trends
plt.errorbar(epsset, sMean1, yerr = np.minimum(np.sqrt(sMean1), np.divide(sMean1, 2)), color = 'blue', marker = 'o', label = "Small KLD + Gauss (samp)")
plt.errorbar(epsset, oMean1, yerr = np.minimum(np.sqrt(oMean1), np.divide(oMean1, 2)), color = 'green', marker = 'x', label = "Small KLD + Gauss (ord)")
plt.errorbar(epsset, sMean2, yerr = np.minimum(np.sqrt(sMean2), np.divide(sMean2, 2)), color = 'orange', marker = 'o', label = "Small KLD + Gauss (samp) mc")
plt.errorbar(epsset, oMean2, yerr = np.minimum(np.sqrt(oMean2), np.divide(oMean2, 2)), color = 'red', marker = 'x', label = "Small KLD + Gauss (ord) mc")
plt.legend(loc = "best")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (start noise)")
plt.title("Effect of epsilon on error of unbiased estimator")
plt.savefig("Synth_eps_small_start_noise.png")
plt.clf()

plt.errorbar(epsset, sMean3, yerr = np.minimum(np.sqrt(sMean3), np.divide(sMean3, 2)), color = 'blueviolet', marker = 'o', label = "Large KLD + Gauss (samp)")
plt.errorbar(epsset, oMean3, yerr = np.minimum(np.sqrt(oMean3), np.divide(oMean3, 2)), color = 'lime', marker = 'x', label = "Large KLD + Gauss (ord)")
plt.errorbar(epsset, sMean4, yerr = np.minimum(np.sqrt(sMean4), np.divide(sMean4, 2)), color = 'gold', marker = 'o', label = "Large KLD + Gauss (samp) mc")
plt.errorbar(epsset, oMean4, yerr = np.minimum(np.sqrt(oMean4), np.divide(oMean4, 2)), color = 'pink', marker = 'x', label = "Large KLD + Gauss (ord) mc")
plt.legend(loc = "best")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (start noise)")
plt.title("Effect of epsilon on error of unbiased estimator")
plt.savefig("Synth_eps_large_start_noise.png")
plt.clf()

plt.errorbar(epsset, sMean5, yerr = np.minimum(np.sqrt(sMean5), np.divide(sMean5, 2)), color = 'blue', marker = 'o', label = "Small KLD + Gauss (samp)")
plt.errorbar(epsset, oMean5, yerr = np.minimum(np.sqrt(oMean5), np.divide(oMean5, 2)), color = 'green', marker = 'x', label = "Small KLD + Gauss (ord)")
plt.errorbar(epsset, sMean6, yerr = np.minimum(np.sqrt(sMean6), np.divide(sMean6, 2)), color = 'orange', marker = 'o', label = "Small KLD + Gauss (samp) mc")
plt.errorbar(epsset, oMean6, yerr = np.minimum(np.sqrt(oMean6), np.divide(oMean6, 2)), color = 'red', marker = 'x', label = "Small KLD + Gauss (ord) mc")
plt.legend(loc = "best")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (mid noise)")
plt.title("Effect of epsilon on error of unbiased estimator")
plt.savefig("Synth_eps_small_mid_noise.png")
plt.clf()

plt.errorbar(epsset, sMean7, yerr = np.minimum(np.sqrt(sMean7), np.divide(sMean7, 2)), color = 'blueviolet', marker = 'o', label = "Large KLD + Gauss (samp)")
plt.errorbar(epsset, oMean7, yerr = np.minimum(np.sqrt(oMean7), np.divide(oMean7, 2)), color = 'lime', marker = 'x', label = "Large KLD + Gauss (ord)")
plt.errorbar(epsset, sMean8, yerr = np.minimum(np.sqrt(sMean8), np.divide(sMean8, 2)), color = 'gold', marker = 'o', label = "Large KLD + Gauss (samp) mc")
plt.errorbar(epsset, oMean8, yerr = np.minimum(np.sqrt(oMean8), np.divide(oMean8, 2)), color = 'pink', marker = 'x', label = "Large KLD + Gauss (ord) mc")
plt.legend(loc = "best")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (mid noise)")
plt.title("Effect of epsilon on error of unbiased estimator")
plt.savefig("Synth_eps_large_mid_noise.png")
plt.clf()

plt.errorbar(epsset, sMean9, yerr = np.minimum(np.sqrt(sMean9), np.divide(sMean9, 2)), color = 'blue', marker = 'o', label = "Small KLD + Lap (samp)")
plt.errorbar(epsset, oMean9, yerr = np.minimum(np.sqrt(oMean9), np.divide(oMean9, 2)), color = 'green', marker = 'x', label = "Small KLD + Lap (ord)")
plt.errorbar(epsset, sMean10, yerr = np.minimum(np.sqrt(sMean10), np.divide(sMean10, 2)), color = 'orange', marker = 'o', label = "Small KLD + Lap (samp) mc")
plt.errorbar(epsset, oMean10, yerr = np.minimum(np.sqrt(oMean10), np.divide(oMean10, 2)), color = 'red', marker = 'x', label = "Small KLD + Lap (ord) mc")
plt.legend(loc = "best")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (end noise)")
plt.title("Effect of epsilon on error of unbiased estimator")
plt.savefig("Synth_eps_small_end_noise.png")
plt.clf()

plt.errorbar(epsset, sMean11, yerr = np.minimum(np.sqrt(sMean11), np.divide(sMean11, 2)), color = 'blueviolet', marker = 'o', label = "Large KLD + Lap (samp)")
plt.errorbar(epsset, oMean11, yerr = np.minimum(np.sqrt(oMean11), np.divide(oMean11, 2)), color = 'lime', marker = 'x', label = "Large KLD + Lap (ord)")
plt.errorbar(epsset, sMean12, yerr = np.minimum(np.sqrt(sMean12), np.divide(sMean12, 2)), color = 'gold', marker = 'o', label = "Large KLD + Lap (samp) mc")
plt.errorbar(epsset, oMean12, yerr = np.minimum(np.sqrt(oMean12), np.divide(oMean12, 2)), color = 'pink', marker = 'x', label = "Large KLD + Lap (ord) mc")
plt.legend(loc = "best")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (end noise)")
plt.title("Effect of epsilon on error of unbiased estimator")
plt.savefig("Synth_eps_large_end_noise.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
