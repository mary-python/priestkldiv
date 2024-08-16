"""Modules provide various time-related functions, compute the natural logarithm of a number, parameterise
probability distributions, create static, animated, and interactive visualisations, and work with arrays."""
import time
from math import log
import torch
import torch.distributions as dis
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# initialising start time and seed for random sampling
startTime = time.perf_counter()
SEED_FREQ = 0
torch.manual_seed(SEED_FREQ)
print("\nStarting...")

# lists of the values of C and lambda, as well as the trials that will be explored
Cset = [40, 80, 120, 160, 200, 260, 320, 400, 480, 560, 620, 680, 740, 800]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
          1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
trialset = ["Dist_C_a", "TAgg_C_a", "Trusted_C_a", "Dist_C_b", "TAgg_C_b", "Trusted_C_b", "no_privacy_C_a", "no_privacy_C_b"]
CS = len(Cset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanEstMSE = np.zeros((TS, CS))
meanPerc = np.zeros((TS, CS))
meanEstRange = np.zeros((TS, CS))
meanPercRange = np.zeros((TS, CS))

# global parameters
T = 500_000 # sample T points
C = 200 # number of clients
N = 125 # number of points per client
EPS = 0.5
DTA = 0.1
A = 0 # parameter for addition of noise
R = 10
RS = 10
SEED_FREQ = 0

for trial in range(8):

    # p is unknown distribution, q is known
    # distributions have small KL divergence
    if trial < 3 or trial == 6:
        p = dis.Laplace(loc = 0.1, scale = 1)

    # distributions have large KL divergence
    else:
        p = dis.Laplace(loc = 1, scale = 1)

    q = dis.Normal(loc = 0, scale = 1)

    # error at end = (output of algorithm - ground truth) squared
    groundTruth = torch.distributions.kl.kl_divergence(p, q)

    # round to 2 d.p., find indices of and eliminate unique values
    pSample = p.sample(sample_shape = (T,))
    pRound = torch.round(pSample, decimals = 2)
    pUnique = torch.unique(pRound, return_counts = True)
    pIndices = (pUnique[1] == 1).nonzero().flatten()
    pUniqueIndices = pUnique[0][pIndices]

    qSample = q.sample(sample_shape = (T,))
    qRound = torch.round(qSample, decimals = 2)
    qUnique = torch.unique(qRound, return_counts = True)
    qIndices = (qUnique[1] == 1).nonzero().flatten()
    qUniqueIndices = qUnique[0][qIndices]

    for i in pUniqueIndices:
        pRound = pRound[pRound != i]

    for i in qUniqueIndices:
        qRound = qRound[qRound != i]

    # order the pre-processed sample
    pOrderedRound = torch.sort(pRound)
    qOrderedRound = torch.sort(qRound)

    if trial < 6:
        b1 = 1 + log(2)
        b2 = 2*((log(1.25))/DTA)*b1
    
        # load Laplace and Gaussian noise distributions, dependent on eps
        s1 = b1 / EPS
        s2 = b2 / EPS
        s3 = s2 * (np.sqrt(2) / R)

        if trial % 3 == 0:
            probGaussNoise = dis.Normal(loc = A, scale = s3 / 100)
        elif trial % 3 == 1:
            gaussNoise = dis.Normal(loc = A, scale = s3)   
        else:
            lapNoise = dis.Laplace(loc = A, scale = s1)

    C_COUNT = 0

    for C in Cset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        tempMeanEst = np.zeros(RS)
        tempMeanEstMSE = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        
        for rep in range(RS):
            print(f"C = {C}, repeat = {rep + 1}...")
        
            # initialising seed for random sampling
            torch.manual_seed(SEED_FREQ)

            meanRangeEst = np.zeros((LS, C))
            startNoise = []

            for j in range(C):

                # each client gets N points from pre-processed sample
                qClientSamp = qOrderedRound[0][N*j : N*(j + 1)]
                logp = p.log_prob(qClientSamp)
                logq = q.log_prob(qClientSamp)

                # "Dist" (each client adds Gaussian noise term)
                if trial % 3 == 0 and trial != 6:
                    startSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                    startNoise.append(startSample)
                    logp = logp + log(startSample)

                # compute ratio between unknown and known distributions
                logr = abs(logp - logq)
                r = np.exp(logr)

                LDA_COUNT = 0

                # explore lambdas in a range
                for lda in ldaset:

                    # compute k3 estimator
                    rangeEst = lda * (r - 1) - logr

                    # share PRIEST-KLD with server
                    meanRangeEst[LDA_COUNT, j] = rangeEst.mean()
                    LDA_COUNT = LDA_COUNT + 1

            # compute mean of PRIEST-KLD across clients
            meanLda = np.mean(meanRangeEst, axis = 1)
            meanLdaNoise = np.zeros(LS)

            for l in range(LS):

                # "TAgg" (intermediate server adds Gaussian noise term)
                if trial % 3 == 1 and trial != 7:
                    meanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    meanLda[l] = meanLda[l] + meanLdaNoise[l]

            # choose best lambda from experiment 1
            ldaIndex = 10

            # mean across clients for best lambda
            tempMeanEst[rep] = meanLda[ldaIndex]
            
            # "Trusted" (server adds Laplace noise term to final result)
            if trial % 3 == 2:
                meanNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEstMSE[rep] = (tempMeanEst[rep] + meanNoise - groundTruth)**2

            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEstMSE[rep] = (tempMeanEst[rep] - groundTruth)**2

            # compute % of noise vs ground truth
            if trial % 3 == 0 and trial != 6:
                tempMeanPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise) + groundTruth))))*100
                startNoise = [float(sn) for sn in startNoise]

            if trial % 3 == 1 and trial != 7:
                tempMeanPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + groundTruth))*100
                meanLdaNoise = [float(mln) for mln in meanLdaNoise]

            if trial % 3 == 2:
                tempMeanPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise + groundTruth))))*100
                meanNoise = [float(mn) for mn in meanNoise]

            SEED_FREQ = SEED_FREQ + 1
        
        # compute mean of repeats
        meanEstMSE[trial, C_COUNT] = np.mean(tempMeanEstMSE)
        meanPerc[trial, C_COUNT] = np.mean(tempMeanPerc)
        
        # compute standard deviation of repeats
        meanEstRange[trial, C_COUNT] = np.std(tempMeanEstMSE)
        meanPercRange[trial, C_COUNT] = np.std(tempMeanPerc)

        C_COUNT = C_COUNT + 1

# EXPERIMENT 2: MSE of PRIEST-KLD for each C
plt.errorbar(Cset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstMSE[6], yerr = np.minimum(meanEstRange[6], np.sqrt(meanEstMSE[6]), np.divide(meanEstMSE[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = "best")
plt.yscale('log')
plt.ylim(0.2, 100)
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_synth_C_est_a.png")
plt.clf()

plt.errorbar(Cset, meanEstMSE[3], yerr = np.minimum(meanEstRange[3], np.sqrt(meanEstMSE[3]), np.divide(meanEstMSE[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstMSE[4], yerr = np.minimum(meanEstRange[4], np.sqrt(meanEstMSE[4]), np.divide(meanEstMSE[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstMSE[5], yerr = np.minimum(meanEstRange[5], np.sqrt(meanEstMSE[5]), np.divide(meanEstMSE[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstMSE[7], yerr = np.minimum(meanEstRange[7], np.sqrt(meanEstMSE[7]), np.divide(meanEstMSE[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = "best")
plt.yscale('log')
plt.yticks([1, 10, 60])
plt.ylim(1, 60)
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_synth_C_est_b.png")
plt.clf()

# EXPERIMENT 3: % of noise vs ground truth for each epsilon
plt.errorbar(Cset, meanPerc[0], yerr = np.minimum(meanPercRange[0], np.sqrt(meanPerc[0]), np.divide(meanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanPerc[1], yerr = np.minimum(meanPercRange[1], np.sqrt(meanPerc[1]), np.divide(meanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanPerc[2], yerr = np.minimum(meanPercRange[2], np.sqrt(meanPerc[2]), np.divide(meanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanPerc[6], yerr = np.minimum(meanPercRange[6], np.sqrt(meanPerc[6]), np.divide(meanPerc[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("Noise (%)")
plt.savefig("Exp3_synth_C_perc_a.png")
plt.clf()

plt.errorbar(Cset, meanPerc[3], yerr = np.minimum(meanPercRange[3], np.sqrt(meanPerc[3]), np.divide(meanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanPerc[4], yerr = np.minimum(meanPercRange[4], np.sqrt(meanPerc[4]), np.divide(meanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanPerc[5], yerr = np.minimum(meanPercRange[5], np.sqrt(meanPerc[5]), np.divide(meanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanPerc[7], yerr = np.minimum(meanPercRange[7], np.sqrt(meanPerc[7]), np.divide(meanPerc[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("Noise (%)")
plt.savefig("Exp3_synth_C_perc_b.png")
plt.clf()

# compute total runtime
totalTime = time.perf_counter() - startTime
hours = totalTime // 3600
minutes = (totalTime % 3600) // 60
seconds = totalTime % 60

if hours == 0:
    if minutes == 1:
        print(f"\nRuntime: {round(minutes)} minute {round(seconds, 2)} seconds.\n")
    else:
        print(f"\nRuntime: {round(minutes)} minutes {round(seconds, 2)} seconds.\n")
elif hours == 1:
    if minutes == 1:
        print(f"\nRuntime: {round(hours)} hour {round(minutes)} minute {round(seconds, 2)} seconds.\n")
    else:
        print(f"\nRuntime: {round(hours)} hour {round(minutes)} minutes {round(seconds, 2)} seconds.\n")
else:
    if minutes == 1:
        print(f"\nRuntime: {round(hours)} hours {round(minutes)} minute {round(seconds, 2)} seconds.\n")
    else:
        print(f"\nRuntime: {round(hours)} hours {round(minutes)} minutes {round(seconds, 2)} seconds.\n")