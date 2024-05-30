"""Modules provide various time-related functions, compute the natural logarithm of a number, parameterise
probability distributions, create static, animated, and interactive visualisations, and work with arrays."""
import time
from math import log
import torch
import torch.distributions as dis
import matplotlib.pyplot as plt
import numpy as np

# initialising start time and seed for random sampling
startTime = time.perf_counter()
print("\nStarting...")
torch.manual_seed(12)

# lists of the values of epsilon and trials that will be run
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
trialset = ["small_start_gauss", "small_start_gauss_mc", "small_mid_gauss", "small_mid_gauss_mc", "small_end_lap", "small_end_lap_mc",
            "large_start_gauss", "large_start_gauss_mc", "large_mid_gauss", "large_mid_gauss_mc", "large_end_lap", "large_end_lap_mc"]
ES = len(epsset)
TS = len(trialset)

# stores for mean of unbiased estimator and optimum lambda
sMeanEst = np.zeros((TS, ES))
sLdaOpt = np.zeros((TS, ES))
oMeanEst = np.zeros((TS, ES))
oLdaOpt = np.zeros((TS, ES))

for trial in range(12):
    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    statsfile = open(f"synth_{trialset[trial]}.txt", "w", encoding = 'utf-8')

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial < 6:
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
    EPS_COUNT = 0

    for eps in epsset:
        print(f"Trial {trial + 1}: epsilon = {eps}...")

        # load Gaussian and Laplace noise distributions, dependent on eps
        s1 = b1 / eps
        s2 = b2 / eps

        if trial % 3 == 0 or trial % 3 == 1:
            s3 = s2 * (np.sqrt(2) / R)

            if trial % 3 == 0:
                probGaussNoise = dis.Normal(loc = A, scale = s3 / 100)
            else:
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

            # "SAMPLED": even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                indices = torch.randperm(len(qPositiveRound))[:N]
                qClientSamp = qPositiveRound[indices]
            else:
                indices = torch.randperm(len(qNegativeRound))[:N]
                qClientSamp = qNegativeRound[indices]

            # "ORDERED": each client gets N points in order from ordered pre-processed sample
            qOrdClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

            # option 3a: each client adds Gaussian noise term
            if trial % 3 == 0:
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
        
        # option 3b: intermediate server adds Gaussian noise term
        if trial % 3 == 1:
            for l in range(L):
                sMeanLda[l] = sMeanLda[l] + gaussNoise.sample(sample_shape = (1,))
                oMeanLda[l] = oMeanLda[l] + gaussNoise.sample(sample_shape = (1,))

        # find lambda that produces minimum error
        sLdaIndex = np.argmin(sMeanLda)
        oLdaIndex = np.argmin(oMeanLda)

        sMinMeanError = sMeanLda[sLdaIndex]
        oMinMeanError = oMeanLda[oLdaIndex]

        # mean across clients for optimum lambda
        sMeanEst[trial, EPS_COUNT] = sMinMeanError
        oMeanEst[trial, EPS_COUNT] = oMinMeanError

        sLdaOpt[trial, EPS_COUNT] = sLdaIndex * ldaStep
        oLdaOpt[trial, EPS_COUNT] = oLdaIndex * ldaStep

        # option 3c: server adds Laplace noise term to final result
        if trial % 3 == 2:
            sMeanEst[trial, EPS_COUNT] = (sMeanEst[trial, EPS_COUNT] + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2

        # option 3a/b: intermediate server has already added Gaussian noise term
        else:
            sMeanEst[trial, EPS_COUNT] = (sMeanEst[trial, EPS_COUNT] - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] - groundTruth)**2

        statsfile.write(f"SYNTHETIC: Eps = {eps}\n")
        statsfile.write(f"Sampled: Optimal Lambda {round(sLdaOpt[trial, EPS_COUNT], 2)} for Mean Error {round(sMeanEst[trial, EPS_COUNT], 2)}\n")
        statsfile.write(f"Ordered: Optimal Lambda {round(oLdaOpt[trial, EPS_COUNT], 2)} for Mean Error {round(oMeanEst[trial, EPS_COUNT], 2)}\n\n")

        EPS_COUNT = EPS_COUNT + 1

# Small KLD, sampled
plt.errorbar(epsset, sMeanEst[0], yerr = np.minimum(np.sqrt(sMeanEst[0]), np.divide(sMeanEst[0], 2)), color = 'blue', marker = 'o', label = "start gauss")
plt.errorbar(epsset, sMeanEst[1], yerr = np.minimum(np.sqrt(sMeanEst[1]), np.divide(sMeanEst[1], 2)), color = 'blueviolet', marker = 'x', label = "start gauss mc")
plt.errorbar(epsset, sMeanEst[2], yerr = np.minimum(np.sqrt(sMeanEst[2]), np.divide(sMeanEst[2], 2)), color = 'green', marker = 'o', label = "mid gauss")
plt.errorbar(epsset, sMeanEst[3], yerr = np.minimum(np.sqrt(sMeanEst[3]), np.divide(sMeanEst[3], 2)), color = 'lime', marker = 'x', label = "mid gauss mc")
plt.errorbar(epsset, sMeanEst[4], yerr = np.minimum(np.sqrt(sMeanEst[4]), np.divide(sMeanEst[4], 2)), color = 'orange', marker = 'o', label = "end lap")
plt.errorbar(epsset, sMeanEst[5], yerr = np.minimum(np.sqrt(sMeanEst[5]), np.divide(sMeanEst[5], 2)), color = 'gold', marker = 'x', label = "end lap mc")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (small KLD, samp)")
plt.title("Effect of epsilon on error of unbiased estimator (small KLD, samp)")
plt.savefig("Synth_eps_small_samp.png")
plt.clf()

plt.plot(epsset, sLdaOpt[0], color = 'blue', marker = 'o', label = 'start gauss')
plt.plot(epsset, sLdaOpt[1], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plt.plot(epsset, sLdaOpt[2], color = 'green', marker = 'o', label = 'mid gauss')
plt.plot(epsset, sLdaOpt[3], color = 'lime', marker = 'x', label = 'mid gauss mc')
plt.plot(epsset, sLdaOpt[4], color = 'orange', marker = 'o', label = 'end lap')
plt.plot(epsset, sLdaOpt[5], color = 'gold', marker = 'x', label = 'end lap mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator (small KLD, samp)")
plt.title("How epsilon affects optimum lambda (small KLD, samp)")
plt.savefig("Synth_eps_small_samp_lda.png")
plt.clf()

# Small KLD, ordered
plt.errorbar(epsset, oMeanEst[0], yerr = np.minimum(np.sqrt(oMeanEst[0]), np.divide(oMeanEst[0], 2)), color = 'blue', marker = 'o', label = "start gauss")
plt.errorbar(epsset, oMeanEst[1], yerr = np.minimum(np.sqrt(oMeanEst[1]), np.divide(oMeanEst[1], 2)), color = 'blueviolet', marker = 'x', label = "start gauss mc")
plt.errorbar(epsset, oMeanEst[2], yerr = np.minimum(np.sqrt(oMeanEst[2]), np.divide(oMeanEst[2], 2)), color = 'green', marker = 'o', label = "mid gauss")
plt.errorbar(epsset, oMeanEst[3], yerr = np.minimum(np.sqrt(oMeanEst[3]), np.divide(oMeanEst[3], 2)), color = 'lime', marker = 'x', label = "mid gauss mc")
plt.errorbar(epsset, oMeanEst[4], yerr = np.minimum(np.sqrt(oMeanEst[4]), np.divide(oMeanEst[4], 2)), color = 'orange', marker = 'o', label = "end lap")
plt.errorbar(epsset, oMeanEst[5], yerr = np.minimum(np.sqrt(oMeanEst[5]), np.divide(oMeanEst[5], 2)), color = 'gold', marker = 'x', label = "end lap mc")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (small KLD, ord)")
plt.title("Effect of epsilon on error of unbiased estimator (small KLD, ord)")
plt.savefig("Synth_eps_small_ord.png")
plt.clf()

plt.plot(epsset, oLdaOpt[0], color = 'blue', marker = 'o', label = 'start gauss')
plt.plot(epsset, oLdaOpt[1], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plt.plot(epsset, oLdaOpt[2], color = 'green', marker = 'o', label = 'mid gauss')
plt.plot(epsset, oLdaOpt[3], color = 'lime', marker = 'x', label = 'mid gauss mc')
plt.plot(epsset, oLdaOpt[4], color = 'orange', marker = 'o', label = 'end lap')
plt.plot(epsset, oLdaOpt[5], color = 'gold', marker = 'x', label = 'end lap mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator (small KLD, ord)")
plt.title("How epsilon affects optimum lambda (small KLD, ord)")
plt.savefig("Synth_eps_small_ord_lda.png")
plt.clf()

# Large KLD, sampled
plt.errorbar(epsset, sMeanEst[6], yerr = np.minimum(np.sqrt(sMeanEst[6]), np.divide(sMeanEst[6], 2)), color = 'blue', marker = 'o', label = "start gauss")
plt.errorbar(epsset, sMeanEst[7], yerr = np.minimum(np.sqrt(sMeanEst[7]), np.divide(sMeanEst[7], 2)), color = 'blueviolet', marker = 'x', label = "start gauss mc")
plt.errorbar(epsset, sMeanEst[8], yerr = np.minimum(np.sqrt(sMeanEst[8]), np.divide(sMeanEst[8], 2)), color = 'green', marker = 'o', label = "mid gauss")
plt.errorbar(epsset, sMeanEst[9], yerr = np.minimum(np.sqrt(sMeanEst[9]), np.divide(sMeanEst[9], 2)), color = 'lime', marker = 'x', label = "mid gauss mc")
plt.errorbar(epsset, sMeanEst[10], yerr = np.minimum(np.sqrt(sMeanEst[10]), np.divide(sMeanEst[10], 2)), color = 'orange', marker = 'o', label = "end lap")
plt.errorbar(epsset, sMeanEst[11], yerr = np.minimum(np.sqrt(sMeanEst[11]), np.divide(sMeanEst[11], 2)), color = 'gold', marker = 'x', label = "end lap mc")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (large KLD, samp)")
plt.title("Effect of epsilon on error of unbiased estimator (large KLD, samp)")
plt.savefig("Synth_eps_large_samp.png")
plt.clf()

plt.plot(epsset, sLdaOpt[6], color = 'blue', marker = 'o', label = 'start gauss')
plt.plot(epsset, sLdaOpt[7], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plt.plot(epsset, sLdaOpt[8], color = 'green', marker = 'o', label = 'mid gauss')
plt.plot(epsset, sLdaOpt[9], color = 'lime', marker = 'x', label = 'mid gauss mc')
plt.plot(epsset, sLdaOpt[10], color = 'orange', marker = 'o', label = 'end lap')
plt.plot(epsset, sLdaOpt[11], color = 'gold', marker = 'x', label = 'end lap mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator (large KLD, samp)")
plt.title("How epsilon affects optimum lambda (large KLD, samp)")
plt.savefig("Synth_eps_large_samp_lda.png")
plt.clf()

# Large KLD, ordered
plt.errorbar(epsset, oMeanEst[6], yerr = np.minimum(np.sqrt(oMeanEst[6]), np.divide(oMeanEst[6], 2)), color = 'blue', marker = 'o', label = "start gauss")
plt.errorbar(epsset, oMeanEst[7], yerr = np.minimum(np.sqrt(oMeanEst[7]), np.divide(oMeanEst[7], 2)), color = 'blueviolet', marker = 'x', label = "start gauss mc")
plt.errorbar(epsset, oMeanEst[8], yerr = np.minimum(np.sqrt(oMeanEst[8]), np.divide(oMeanEst[8], 2)), color = 'green', marker = 'o', label = "mid gauss")
plt.errorbar(epsset, oMeanEst[9], yerr = np.minimum(np.sqrt(oMeanEst[9]), np.divide(oMeanEst[9], 2)), color = 'lime', marker = 'x', label = "mid gauss mc")
plt.errorbar(epsset, oMeanEst[10], yerr = np.minimum(np.sqrt(oMeanEst[10]), np.divide(oMeanEst[10], 2)), color = 'orange', marker = 'o', label = "end lap")
plt.errorbar(epsset, oMeanEst[11], yerr = np.minimum(np.sqrt(oMeanEst[11]), np.divide(oMeanEst[11], 2)), color = 'gold', marker = 'x', label = "end lap mc")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (large KLD, ord)")
plt.title("Effect of epsilon on error of unbiased estimator (large KLD, ord)")
plt.savefig("Synth_eps_large_ord.png")
plt.clf()

plt.plot(epsset, oLdaOpt[6], color = 'blue', marker = 'o', label = 'start gauss')
plt.plot(epsset, oLdaOpt[7], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plt.plot(epsset, oLdaOpt[8], color = 'green', marker = 'o', label = 'mid gauss')
plt.plot(epsset, oLdaOpt[9], color = 'lime', marker = 'x', label = 'mid gauss mc')
plt.plot(epsset, oLdaOpt[10], color = 'orange', marker = 'o', label = 'end lap')
plt.plot(epsset, oLdaOpt[11], color = 'gold', marker = 'x', label = 'end lap mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator (large KLD, samp)")
plt.title("How epsilon affects optimum lambda (large KLD, samp)")
plt.savefig("Synth_eps_large_ord_lda.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
