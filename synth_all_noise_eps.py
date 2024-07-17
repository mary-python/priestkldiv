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
print("\nStarting...")

# lists of the values of epsilon and lambda, as well as the trials that will be explored
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
          1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
trialset = ["Dist_small", "TAgg_small", "Trusted_small", "Dist_large", "TAgg_large", "Trusted_large", "no_privacy_small", "no_privacy_large"]
ES = len(epsset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanEst = np.zeros((TS, ES))
meanLdaOpt = np.zeros((TS, ES))
meanEstZero = np.zeros((TS, ES))
meanEstOne = np.zeros((TS, ES))
meanPerc = np.zeros((TS, ES))
meanEpsSmall = np.zeros((TS, LS))
meanEpsDef = np.zeros((TS, LS))
meanEpsMid = np.zeros((TS, LS))
meanEpsLarge = np.zeros((TS, LS))

meanEstRange = np.zeros((TS, ES))
meanLdaOptRange = np.zeros((TS, ES))
meanEstZeroRange = np.zeros((TS, ES))
meanEstOneRange = np.zeros((TS, ES))
meanPercRange = np.zeros((TS, ES))
meanEpsSmallRange = np.zeros((TS, LS))
meanEpsDefRange = np.zeros((TS, LS))
meanEpsMidRange = np.zeros((TS, LS))
meanEpsLargeRange = np.zeros((TS, LS))

# global parameters
T = 500_000 # sample T points
C = 200 # number of clients
N = 125 # number of points per client
DTA = 0.1
A = 0 # parameter for addition of noise
R = 10
ldaStep = 0.05
RS = 10
SEED_FREQ = 0
SMALL_INDEX = 0
DEF_INDEX = 3
MID_INDEX = 6
LARGE_INDEX = 10

for trial in range(8):
    ordfile = open(f"synth_eps_{trialset[trial]}.txt", "w", encoding = 'utf-8')

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
    qSample = q.sample(sample_shape = (T,))
    qRound = torch.round(qSample, decimals = 2)
    qUnique = torch.unique(qRound, return_counts = True)
    qIndices = (qUnique[1] == 1).nonzero().flatten()
    qUniqueIndices = qUnique[0][qIndices]

    for i in qUniqueIndices:
        qRound = qRound[qRound != i]

    # order the pre-processed sample
    qOrderedRound = torch.sort(qRound)
    
    if trial < 6:
        b1 = 1 + log(2)
        b2 = 2*((log(1.25))/DTA)*b1

    EPS_COUNT = 0

    for eps in epsset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        tempMeanEst = np.zeros(RS)
        tempMeanLdaOpt = np.zeros(RS)
        tempMeanEstZero = np.zeros(RS)
        tempMeanEstOne = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        tempMeanEpsSmall = np.zeros((LS, RS))
        tempMeanEpsDef = np.zeros((LS, RS))
        tempMeanEpsMid = np.zeros((LS, RS))
        tempMeanEpsLarge = np.zeros((LS, RS))

        for rep in range(RS):
            print(f"epsilon = {eps}, repeat = {rep + 1}...")

            # initialising seed for random sampling
            torch.manual_seed(SEED_FREQ)

            if trial < 6:

                # load Gaussian and Laplace noise distributions, dependent on eps
                s1 = b1 / eps
                s2 = b2 / eps
                s3 = s2 * (np.sqrt(2) / R)

                if trial % 3 == 0:
                    probGaussNoise = dis.Normal(loc = A, scale = s3 / 100)
                elif trial % 3 == 1:
                    gaussNoise = dis.Normal(loc = A, scale = s3)   
                else:
                    lapNoise = dis.Laplace(loc = A, scale = s1)

            meanEst = np.zeros((LS, C))
            startNoise = []

            for j in range(C):

                # "ORDERED": each client gets N points in order fro pre-processed sample
                qClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

                # compute ratio between unknown and known distributions
                logr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)

                # "Dist" (each client adds Gaussian noise term)
                if trial % 3 == 0 and trial != 6:
                    startSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                    startNoise.append(startSample)
                    logr = logr + startSample

                LDA_COUNT = 0

                # explore lambdas in a range
                for lda in ldaset:

                    # compute k3 estimator
                    rangeEst = lda * (np.exp(logr) - 1) - logr

                    # share PRIEST-KLD with server
                    meanEst[LDA_COUNT, j] = rangeEst.mean()
                    LDA_COUNT = LDA_COUNT + 1

            # compute mean of PRIEST-KLD across clients
            meanLda = np.mean(meanEst, axis = 1)
            meanLdaNoise = np.zeros(LS)
        
            for l in range(LS):

                # "TAgg" (intermediate server adds Gaussian noise term)
                if trial % 3 == 1 and trial != 7:
                    meanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    meanLda[l] = meanLda[l] + meanLdaNoise[l]
            
                # mean across lambdas for eps = 0.05 (small)
                if EPS_COUNT == SMALL_INDEX:
                    tempMeanEpsSmall[l, rep] = meanLda[l]

                # eps = 0.5 (default)
                if EPS_COUNT == DEF_INDEX:
                    tempMeanEpsDef[l, rep] = meanLda[l]

                # eps = 1.5 (mid)
                if EPS_COUNT == MID_INDEX:
                    tempMeanEpsMid[l, rep] = meanLda[l]

                # eps = 3 (large)
                if EPS_COUNT == LARGE_INDEX:
                    tempMeanEpsLarge[l, rep] = meanLda[l]

            # find lambda that produces minimum error
            ldaIndex = np.argmin(meanLda)
            minMeanError = meanLda[ldaIndex]

            # mean across clients for optimum lambda
            tempMeanEst[rep] = minMeanError

            # optimum lambda
            tempMeanLdaOpt[rep] = ldaIndex * ldaStep

            # lambda = 0
            tempMeanEstZero[rep] = meanLda[0]

            # lambda = 1
            tempMeanEstOne[rep] = meanLda[LS-1]

            # "Trusted" (server adds Laplace noise term to final result)
            if trial % 3 == 2:
                meanNoise = lapNoise.sample(sample_shape = (1,))
                meanZeroNoise = lapNoise.sample(sample_shape = (1,))
                meanOneNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEst[rep] = (tempMeanEst[rep] + meanNoise - groundTruth)**2
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] + meanZeroNoise - groundTruth)**2
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] + meanOneNoise - groundTruth)**2

                for l in range(LS):
            
                    # eps = 0.05 (small)
                    if EPS_COUNT == 0:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsSmall[l, rep] = (tempMeanEpsSmall[l, rep] + meanSmallNoise - groundTruth)**2

                    # eps = 0.5 (def)
                    if EPS_COUNT == 3:
                        meanDefNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsDef[l, rep] = (tempMeanEpsDef[l, rep] + meanDefNoise - groundTruth)**2

                    # eps = 1.5 (mid)
                    if EPS_COUNT == 6:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsMid[l, rep] = (tempMeanEpsMid[l, rep] + meanMidNoise - groundTruth)**2

                    # eps = 3 (large)
                    if EPS_COUNT == 10:
                        meanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsLarge[l, rep] = (tempMeanEpsLarge[l, rep] + meanLargeNoise - groundTruth)**2

            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEst[rep] = (tempMeanEst[rep] - groundTruth)**2
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] - groundTruth)**2
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] - groundTruth)**2

                for l in range(LS):
            
                    # eps = 0.05 (small)
                    if EPS_COUNT == 0:
                        tempMeanEpsSmall[l, rep] = (tempMeanEpsSmall[l, rep] - groundTruth)**2

                    # eps = 0.5 (def)
                    if EPS_COUNT == 3:
                        tempMeanEpsDef[l, rep] = (tempMeanEpsDef[l, rep] - groundTruth)**2

                    # eps = 1.5 (mid)
                    if EPS_COUNT == 6:
                        tempMeanEpsMid[l, rep] = (tempMeanEpsMid[l, rep] - groundTruth)**2

                    # eps = 3 (large)
                    if EPS_COUNT == 10:
                        tempMeanEpsLarge[l, rep] = (tempMeanEpsLarge[l, rep] - groundTruth)**2

            # compute % of noise vs ground truth
            if trial % 3 == 0 and trial != 6:
                tempMeanPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise) + groundTruth))))*100
            if trial % 3 == 1 and trial != 7:
                tempMeanPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + groundTruth))*100
            if trial % 3 == 2:
                tempMeanPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise + groundTruth))))*100
        
            SEED_FREQ = SEED_FREQ + 1

        # compute mean of repeats
        meanEst[trial, EPS_COUNT] = np.mean(tempMeanEst)
        meanLdaOpt[trial, EPS_COUNT] = np.mean(tempMeanLdaOpt)
        meanEstZero[trial, EPS_COUNT] = np.mean(tempMeanEstZero)
        meanEstOne[trial, EPS_COUNT] = np.mean(tempMeanEstOne)
        meanPerc[trial, EPS_COUNT] = np.mean(tempMeanPerc)

        for l in range(LS):
            if EPS_COUNT == SMALL_INDEX:
                meanEpsSmall[trial, l] = np.mean(tempMeanEpsSmall[l])
            if EPS_COUNT == DEF_INDEX:
                meanEpsDef[trial, l] = np.mean(tempMeanEpsDef[l])
            if EPS_COUNT == MID_INDEX:
                meanEpsMid[trial, l] = np.mean(tempMeanEpsMid[l])
            if EPS_COUNT == LARGE_INDEX:
                meanEpsLarge[trial, l] = np.mean(tempMeanEpsLarge[l])
        
        # compute standard deviation of repeats
        meanEstRange[trial, EPS_COUNT] = np.std(tempMeanEst)
        meanLdaOptRange[trial, EPS_COUNT] = np.std(tempMeanLdaOpt)
        meanEstZeroRange[trial, EPS_COUNT] = np.std(tempMeanEstZero)
        meanEstOneRange[trial, EPS_COUNT] = np.std(tempMeanEstOne)
        meanPercRange[trial, EPS_COUNT] = np.std(tempMeanPerc)

        for l in range(LS):
            if EPS_COUNT == SMALL_INDEX:
                meanEpsSmallRange[trial, l] = np.std(tempMeanEpsSmall[l])
            if EPS_COUNT == DEF_INDEX:
                meanEpsDefRange[trial, l] = np.std(tempMeanEpsDef[l])
            if EPS_COUNT == MID_INDEX:
                meanEpsMidRange[trial, l] = np.std(tempMeanEpsMid[l])
            if EPS_COUNT == LARGE_INDEX:
                meanEpsLargeRange[trial, l] = np.std(tempMeanEpsLarge[l])

        # write statistics on data files
        if eps == epsset[0]:
            ordfile.write(f"SYNTHETIC Ordered: Eps = {eps}\n")
        else:
            ordfile.write(f"\nEps = {eps}\n")

        ordfile.write(f"\nMean Error: {round(meanEst[trial, EPS_COUNT], 2)}\n")
        ordfile.write(f"Optimal Lambda: {round(meanLdaOpt[trial, EPS_COUNT], 2)}\n")
        ordfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        ordfile.write(f"Noise: {np.round(meanPerc[trial, EPS_COUNT], 2)}%\n")

        EPS_COUNT = EPS_COUNT + 1

# plot error of PRIEST-KLD for each epsilon (small KLD)
plt.errorbar(epsset, meanEst[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEst[0]), np.divide(meanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEst[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEst[1]), np.divide(meanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEst[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEst[2]), np.divide(meanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD)
plt.errorbar(epsset, meanEst[3], yerr = np.minimum(meanEstRange[3], np.sqrt(meanEst[3]), np.divide(meanEst[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEst[4], yerr = np.minimum(meanEstRange[4], np.sqrt(meanEst[4]), np.divide(meanEst[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEst[5], yerr = np.minimum(meanEstRange[5], np.sqrt(meanEst[5]), np.divide(meanEst[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD)
plt.errorbar(epsset, meanEstZero[0], yerr = np.minimum(meanEstZeroRange[0], np.sqrt(meanEstZero[0]), np.divide(meanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstZero[1], yerr = np.minimum(meanEstZeroRange[1], np.sqrt(meanEstZero[1]), np.divide(meanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstZero[2], yerr = np.minimum(meanEstZeroRange[2], np.sqrt(meanEstZero[2]), np.divide(meanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD)
plt.errorbar(epsset, meanEstZero[3], yerr = np.minimum(meanEstZeroRange[3], np.sqrt(meanEstZero[3]), np.divide(meanEstZero[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstZero[4], yerr = np.minimum(meanEstZeroRange[4], np.sqrt(meanEstZero[4]), np.divide(meanEstZero[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstZero[5], yerr = np.minimum(meanEstZeroRange[5], np.sqrt(meanEstZero[5]), np.divide(meanEstZero[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD)
plt.errorbar(epsset, meanEstOne[0], yerr = np.minimum(meanEstOneRange[0], np.sqrt(meanEstOne[0]), np.divide(meanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstOne[1], yerr = np.minimum(meanEstOneRange[1], np.sqrt(meanEstOne[1]), np.divide(meanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstOne[2], yerr = np.minimum(meanEstOneRange[2], np.sqrt(meanEstOne[2]), np.divide(meanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD)
plt.errorbar(epsset, meanEstOne[3], yerr = np.minimum(meanEstOneRange[3], np.sqrt(meanEstOne[3]), np.divide(meanEstOne[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstOne[4], yerr = np.minimum(meanEstOneRange[4], np.sqrt(meanEstOne[4]), np.divide(meanEstOne[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstOne[5], yerr = np.minimum(meanEstOneRange[5], np.sqrt(meanEstOne[5]), np.divide(meanEstOne[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (small KLD)
plt.errorbar(ldaset, meanEpsSmall[0], yerr = np.minimum(meanEpsSmallRange[0], np.sqrt(meanEpsSmall[0]), np.divide(meanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsSmall[1], yerr = np.minimum(meanEpsSmallRange[1], np.sqrt(meanEpsSmall[1]), np.divide(meanEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsSmall[2], yerr = np.minimum(meanEpsSmallRange[2], np.sqrt(meanEpsSmall[2]), np.divide(meanEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsSmall[6], yerr = np.minimum(meanEpsSmallRange[6], np.sqrt(meanEpsSmall[6]), np.divide(meanEpsSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (large KLD)
plt.errorbar(ldaset, meanEpsSmall[3], yerr = np.minimum(meanEpsSmallRange[3], np.sqrt(meanEpsSmall[3]), np.divide(meanEpsSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsSmall[4], yerr = np.minimum(meanEpsSmallRange[4], np.sqrt(meanEpsSmall[4]), np.divide(meanEpsSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsSmall[5], yerr = np.minimum(meanEpsSmallRange[5], np.sqrt(meanEpsSmall[5]), np.divide(meanEpsSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsSmall[7], yerr = np.minimum(meanEpsSmallRange[7], np.sqrt(meanEpsSmall[7]), np.divide(meanEpsSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (small KLD)
plt.errorbar(ldaset, meanEpsDef[0], yerr = np.minimum(meanEpsDefRange[0], np.sqrt(meanEpsDef[0]), np.divide(meanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsDef[1], yerr = np.minimum(meanEpsDefRange[1], np.sqrt(meanEpsDef[1]), np.divide(meanEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsDef[2], yerr = np.minimum(meanEpsDefRange[2], np.sqrt(meanEpsDef[2]), np.divide(meanEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsDef[6], yerr = np.minimum(meanEpsDefRange[6], np.sqrt(meanEpsDef[6]), np.divide(meanEpsDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (large KLD)
plt.errorbar(ldaset, meanEpsDef[3], yerr = np.minimum(meanEpsDefRange[3], np.sqrt(meanEpsDef[3]), np.divide(meanEpsDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsDef[4], yerr = np.minimum(meanEpsDefRange[4], np.sqrt(meanEpsDef[4]), np.divide(meanEpsDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsDef[5], yerr = np.minimum(meanEpsDefRange[5], np.sqrt(meanEpsDef[5]), np.divide(meanEpsDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsDef[7], yerr = np.minimum(meanEpsDefRange[7], np.sqrt(meanEpsDef[7]), np.divide(meanEpsDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (small KLD)
plt.errorbar(ldaset, meanEpsMid[0], yerr = np.minimum(meanEpsMidRange[0], np.sqrt(meanEpsMid[0]), np.divide(meanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsMid[1], yerr = np.minimum(meanEpsMidRange[1], np.sqrt(meanEpsMid[1]), np.divide(meanEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsMid[2], yerr = np.minimum(meanEpsMidRange[2], np.sqrt(meanEpsMid[2]), np.divide(meanEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsMid[6], yerr = np.minimum(meanEpsMidRange[6], np.sqrt(meanEpsMid[6]), np.divide(meanEpsMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (large KLD)
plt.errorbar(ldaset, meanEpsMid[3], yerr = np.minimum(meanEpsMidRange[3], np.sqrt(meanEpsMid[3]), np.divide(meanEpsMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsMid[4], yerr = np.minimum(meanEpsMidRange[4], np.sqrt(meanEpsMid[4]), np.divide(meanEpsMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsMid[5], yerr = np.minimum(meanEpsMidRange[5], np.sqrt(meanEpsMid[5]), np.divide(meanEpsMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsMid[7], yerr = np.minimum(meanEpsMidRange[7], np.sqrt(meanEpsMid[7]), np.divide(meanEpsMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (small KLD)
plt.errorbar(ldaset, meanEpsLarge[0], yerr = np.minimum(meanEpsLargeRange[0], np.sqrt(meanEpsLarge[0]), np.divide(meanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsLarge[1], yerr = np.minimum(meanEpsLargeRange[1], np.sqrt(meanEpsLarge[1]), np.divide(meanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsLarge[2], yerr = np.minimum(meanEpsLargeRange[2], np.sqrt(meanEpsLarge[2]), np.divide(meanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsLarge[6], yerr = np.minimum(meanEpsLargeRange[6], np.sqrt(meanEpsLarge[6]), np.divide(meanEpsLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (large KLD)
plt.errorbar(ldaset, meanEpsLarge[3], yerr = np.minimum(meanEpsLargeRange[3], np.sqrt(meanEpsLarge[3]), np.divide(meanEpsLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsLarge[4], yerr = np.minimum(meanEpsLargeRange[4], np.sqrt(meanEpsLarge[4]), np.divide(meanEpsLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsLarge[5], yerr = np.minimum(meanEpsLargeRange[5], np.sqrt(meanEpsLarge[5]), np.divide(meanEpsLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsLarge[7], yerr = np.minimum(meanEpsLargeRange[7], np.sqrt(meanEpsLarge[7]), np.divide(meanEpsLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_eps_large.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (small KLD)
plt.errorbar(epsset, meanPerc[0], yerr = np.minimum(meanPercRange[0], np.sqrt(meanPerc[0]), np.divide(meanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanPerc[1], yerr = np.minimum(meanPercRange[1], np.sqrt(meanPerc[1]), np.divide(meanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanPerc[2], yerr = np.minimum(meanPercRange[2], np.sqrt(meanPerc[2]), np.divide(meanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Synth_eps_perc_small.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (large KLD)
plt.errorbar(epsset, meanPerc[3], yerr = np.minimum(meanPercRange[3], np.sqrt(meanPerc[3]), np.divide(meanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanPerc[4], yerr = np.minimum(meanPercRange[4], np.sqrt(meanPerc[4]), np.divide(meanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanPerc[5], yerr = np.minimum(meanPercRange[5], np.sqrt(meanPerc[5]), np.divide(meanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Synth_eps_perc_large.png")
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
