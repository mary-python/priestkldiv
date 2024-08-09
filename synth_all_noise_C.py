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
meanLdaBest = np.zeros((TS, CS))
meanSmallBest = np.zeros((TS, CS))
meanDefBest = np.zeros((TS, CS))
meanMidBest = np.zeros((TS, CS))
meanLargeBest = np.zeros((TS, CS))
meanPerc = np.zeros((TS, CS))
meanCSmall = np.zeros((TS, LS))
meanCDef = np.zeros((TS, LS))
meanCMid = np.zeros((TS, LS))
meanCLarge = np.zeros((TS, LS))

meanEstRange = np.zeros((TS, CS))
meanSmallBestRange = np.zeros((TS, CS))
meanSmallLargeRange = np.zeros((TS, CS))
meanDefBestRange = np.zeros((TS, CS))
meanMidBestRange = np.zeros((TS, CS))
meanLargeBestRange = np.zeros((TS, CS))
meanCSmallRange = np.zeros((TS, LS))
meanCDefRange = np.zeros((TS, LS))
meanCMidRange = np.zeros((TS, LS))
meanCLargeRange = np.zeros((TS, LS))

# global parameters
T = 500_000 # sample T points
C = 200 # number of clients
N = 125 # number of points per client
EPS = 0.5
DTA = 0.1
A = 0 # parameter for addition of noise
R = 10
ldaStep = 0.05
RS = 10
SEED_FREQ = 0
SMALL_INDEX = 0
DEF_INDEX = 4
MID_INDEX = 7
LARGE_INDEX = 13

for trial in range(8):
    datafile = open(f"Data_{trialset[trial]}.txt", "w", encoding = 'utf-8')

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

    if trial < 6:
    
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
        tempMeanLdaBest = np.zeros(RS)
        tempMeanSmallBest = np.zeros(RS)
        tempMeanSmallLarge = np.zeros(RS)
        tempMeanDefBest = np.zeros(RS)
        tempMeanDefLarge = np.zeros(RS)
        tempMeanMidBest = np.zeros(RS)
        tempMeanMidLarge = np.zeros(RS)
        tempMeanLargeBest = np.zeros(RS)
        tempMeanLargeLarge = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        tempMeanCSmall = np.zeros((LS, RS))
        tempMeanCDef = np.zeros((LS, RS))
        tempMeanCMid = np.zeros((LS, RS))
        tempMeanCLarge = np.zeros((LS, RS))
        
        for rep in range(RS):
            print(f"C = {C}, repeat = {rep + 1}...")
        
            # initialising seed for random sampling
            torch.manual_seed(SEED_FREQ)

            meanRangeEst = np.zeros((LS, C))
            startNoise = []

            for j in range(C):

                # each client gets N points from pre-processed sample
                qClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

                # compute ratio between unknown and known distributions
                logr = abs(p.log_prob(qClientSamp) - q.log_prob(qClientSamp))

                # "Dist" (each client adds Gaussian noise term)
                if trial % 3 == 0 and trial != 6:
                    startSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                    startNoise.append(startSample)
                    logr = logr + log(startSample)

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
            
                # mean across lambdas for C = 40 (small)
                if C_COUNT == SMALL_INDEX:
                    tempMeanCSmall[l, rep] = meanLda[l]

                # C = 200 (default)
                if C_COUNT == DEF_INDEX:
                    tempMeanCDef[l, rep] = meanLda[l]

                # C = 400 (mid)
                if C_COUNT == MID_INDEX:
                    tempMeanCMid[l, rep] = meanLda[l]

                # C = 620 (large)
                if C_COUNT == LARGE_INDEX:
                    tempMeanCLarge[l, rep] = meanLda[l]

            # find lambda that produces minimum error
            ldaIndex = np.argmin(meanLda)
            meanMinError = meanLda[ldaIndex]

            # mean across clients for best lambda
            tempMeanEst[rep] = meanMinError

            # best lambda
            tempMeanLdaBest[rep] = ldaIndex * ldaStep

            # split into small and large KL divergence
            if trial < 3 or trial == 6:

                # C = 40, lambda = 0.5 (Dist)
                tempMeanSmallBest[rep] = meanLda[10]

                # C = 200, lambda = 0.85 (Dist)
                tempMeanDefBest[rep] = meanLda[17]

                # C = 400, lambda = 1.05 (Dist)
                tempMeanMidBest[rep] = meanLda[21]

                # C = 800, lambda = 1.2 (Dist)
                tempMeanLargeBest[rep] = meanLda[24]

            else:

                # C = 40, lambda = 1.5 (Dist)
                tempMeanSmallBest[rep] = meanLda[30]

                # C = 200, lambda = 1.2 (Dist)
                tempMeanDefBest[rep] = meanLda[24]

                # C = 400, lambda = 0.95 (Dist)
                tempMeanMidBest[rep] = meanLda[19]

                # C = 800, lambda = 0.85 (Dist)
                tempMeanLargeBest[rep] = meanLda[17]
            
            # "Trusted" (server adds Laplace noise term to final result)
            if trial % 3 == 2:
                meanNoise = lapNoise.sample(sample_shape = (1,))
                meanSmallBestNoise = lapNoise.sample(sample_shape = (1,))
                meanDefBestNoise = lapNoise.sample(sample_shape = (1,))
                meanMidBestNoise = lapNoise.sample(sample_shape = (1,))
                meanLargeBestNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEstMSE[rep] = (tempMeanEst[rep] + meanNoise - groundTruth)**2
                tempMeanSmallBest[rep] = (tempMeanSmallBest[rep] + meanSmallBestNoise - groundTruth)**2
                tempMeanDefBest[rep] = (tempMeanDefBest[rep] + meanDefBestNoise - groundTruth)**2
                tempMeanMidBest[rep] = (tempMeanMidBest[rep] + meanMidBestNoise - groundTruth)**2
                tempMeanLargeBest[rep] = (tempMeanLargeBest[rep] + meanLargeBestNoise - groundTruth)**2

                for l in range(LS):
            
                    # C = 40 (small)
                    if C_COUNT == SMALL_INDEX:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCSmall[l, rep] = (tempMeanCSmall[l, rep] + meanSmallNoise - groundTruth)**2

                    # C = 200 (def)
                    if C_COUNT == DEF_INDEX:
                        meanDefNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCDef[l, rep] = (tempMeanCDef[l, rep] + meanDefNoise - groundTruth)**2

                    # C = 400 (mid)
                    if C_COUNT == MID_INDEX:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCMid[l, rep] = (tempMeanCMid[l, rep] + meanMidNoise - groundTruth)**2

                    # C = 800 (large)
                    if C_COUNT == LARGE_INDEX:
                        meanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCLarge[l, rep] = (tempMeanCLarge[l, rep] + meanLargeNoise - groundTruth)**2

            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEstMSE[rep] = (tempMeanEst[rep] - groundTruth)**2
                tempMeanSmallBest[rep] = (tempMeanSmallBest[rep] - groundTruth)**2
                tempMeanDefBest[rep] = (tempMeanDefBest[rep] - groundTruth)**2
                tempMeanMidBest[rep] = (tempMeanMidBest[rep] - groundTruth)**2
                tempMeanLargeBest[rep] = (tempMeanLargeBest[rep] - groundTruth)**2

                for l in range(LS):
            
                    # C = 40 (small)
                    if C_COUNT == SMALL_INDEX:
                        tempMeanCSmall[l, rep] = (tempMeanCSmall[l, rep] - groundTruth)**2

                    # C = 200 (def)
                    if C_COUNT == DEF_INDEX:
                        tempMeanCDef[l, rep] = (tempMeanCDef[l, rep] - groundTruth)**2

                    # C = 400 (mid)
                    if C_COUNT == MID_INDEX:
                        tempMeanCMid[l, rep] = (tempMeanCMid[l, rep] - groundTruth)**2

                    # C = 800 (large)
                    if C_COUNT == LARGE_INDEX:
                        tempMeanCLarge[l, rep] = (tempMeanCLarge[l, rep] - groundTruth)**2

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
        meanLdaBest[trial, C_COUNT] = np.mean(tempMeanLdaBest)
        meanSmallBest[trial, C_COUNT] = np.mean(tempMeanSmallBest)
        meanDefBest[trial, C_COUNT] = np.mean(tempMeanDefBest)
        meanMidBest[trial, C_COUNT] = np.mean(tempMeanMidBest)
        meanLargeBest[trial, C_COUNT] = np.mean(tempMeanLargeBest)
        meanPerc[trial, C_COUNT] = np.mean(tempMeanPerc)

        for l in range(LS):
            if C_COUNT == SMALL_INDEX:
                meanCSmall[trial, l] = np.mean(tempMeanCSmall[l])
            if C_COUNT == DEF_INDEX:
                meanCDef[trial, l] = np.mean(tempMeanCDef[l])
            if C_COUNT == MID_INDEX:
                meanCMid[trial, l] = np.mean(tempMeanCMid[l])
            if C_COUNT == LARGE_INDEX:
                meanCLarge[trial, l] = np.mean(tempMeanCLarge[l])
        
        # compute standard deviation of repeats
        meanEstRange[trial, C_COUNT] = np.std(tempMeanEstMSE)
        meanSmallBestRange[trial, C_COUNT] = np.std(tempMeanSmallBest)
        meanDefBestRange[trial, C_COUNT] = np.std(tempMeanDefBest)
        meanMidBestRange[trial, C_COUNT] = np.std(tempMeanMidBest)
        meanLargeBestRange[trial, C_COUNT] = np.std(tempMeanLargeBest)

        for l in range(LS):
            if C_COUNT == SMALL_INDEX:
                meanCSmallRange[trial, l] = np.std(tempMeanCSmall[l])
            if C_COUNT == DEF_INDEX:
                meanCDefRange[trial, l] = np.std(tempMeanCDef[l])
            if C_COUNT == MID_INDEX:
                meanCMidRange[trial, l] = np.std(tempMeanCMid[l])
            if C_COUNT == LARGE_INDEX:
                meanCLargeRange[trial, l] = np.std(tempMeanCLarge[l])

        # write statistics on data files
        if C == Cset[0]:
            datafile.write(f"SYNTHETIC: C = {C}\n")
        else:
            datafile.write(f"\nC = {C}\n")

        datafile.write(f"\nMean MSE: {round(meanEstMSE[trial, C_COUNT], 2)}\n")
        datafile.write(f"Best Lambda: {round(meanLdaBest[trial, C_COUNT], 2)}\n")
        datafile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        datafile.write(f"Noise: {np.round(meanPerc[trial, C_COUNT], 2)}%\n")

        C_COUNT = C_COUNT + 1

# EXPERIMENT 1: MSE of PRIEST-KLD for each C
plt.errorbar(Cset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstMSE[6], yerr = np.minimum(meanEstRange[6], np.sqrt(meanEstMSE[6]), np.divide(meanEstMSE[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = "best")
plt.yscale('log')
plt.ylim(0.2, 100)
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_synth_C_est_a.png")
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
plt.savefig("Exp1_synth_C_est_b.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD for fixed C (40, 200, 400, 800)
plt.errorbar(ldaset, meanCSmall[0], yerr = np.minimum(meanCSmallRange[0], np.sqrt(meanCSmall[0]), np.divide(meanCSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCSmall[1], yerr = np.minimum(meanCSmallRange[1], np.sqrt(meanCSmall[1]), np.divide(meanCSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCSmall[2], yerr = np.minimum(meanCSmallRange[2], np.sqrt(meanCSmall[2]), np.divide(meanCSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCSmall[6], yerr = np.minimum(meanCSmallRange[6], np.sqrt(meanCSmall[6]), np.divide(meanCSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_40_a.png")
plt.clf()

plt.errorbar(ldaset, meanCSmall[3], yerr = np.minimum(meanCSmallRange[3], np.sqrt(meanCSmall[3]), np.divide(meanCSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCSmall[4], yerr = np.minimum(meanCSmallRange[4], np.sqrt(meanCSmall[4]), np.divide(meanCSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCSmall[5], yerr = np.minimum(meanCSmallRange[5], np.sqrt(meanCSmall[5]), np.divide(meanCSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCSmall[7], yerr = np.minimum(meanCSmallRange[7], np.sqrt(meanCSmall[7]), np.divide(meanCSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_40_b.png")
plt.clf()

plt.errorbar(ldaset, meanCDef[0], yerr = np.minimum(meanCDefRange[0], np.sqrt(meanCDef[0]), np.divide(meanCDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCDef[1], yerr = np.minimum(meanCDefRange[1], np.sqrt(meanCDef[1]), np.divide(meanCDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCDef[2], yerr = np.minimum(meanCDefRange[2], np.sqrt(meanCDef[2]), np.divide(meanCDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCDef[6], yerr = np.minimum(meanCDefRange[6], np.sqrt(meanCDef[6]), np.divide(meanCDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_200_a.png")
plt.clf()

plt.errorbar(ldaset, meanCDef[3], yerr = np.minimum(meanCDefRange[3], np.sqrt(meanCDef[3]), np.divide(meanCDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCDef[4], yerr = np.minimum(meanCDefRange[4], np.sqrt(meanCDef[4]), np.divide(meanCDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCDef[5], yerr = np.minimum(meanCDefRange[5], np.sqrt(meanCDef[5]), np.divide(meanCDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCDef[7], yerr = np.minimum(meanCDefRange[7], np.sqrt(meanCDef[7]), np.divide(meanCDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_200_b.png")
plt.clf()

plt.errorbar(ldaset, meanCMid[0], yerr = np.minimum(meanCMidRange[0], np.sqrt(meanCMid[0]), np.divide(meanCMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCMid[1], yerr = np.minimum(meanCMidRange[1], np.sqrt(meanCMid[1]), np.divide(meanCMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCMid[2], yerr = np.minimum(meanCMidRange[2], np.sqrt(meanCMid[2]), np.divide(meanCMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCMid[6], yerr = np.minimum(meanCMidRange[6], np.sqrt(meanCMid[6]), np.divide(meanCMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_400_a.png")
plt.clf()

plt.errorbar(ldaset, meanCMid[3], yerr = np.minimum(meanCMidRange[3], np.sqrt(meanCMid[3]), np.divide(meanCMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCMid[4], yerr = np.minimum(meanCMidRange[4], np.sqrt(meanCMid[4]), np.divide(meanCMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCMid[5], yerr = np.minimum(meanCMidRange[5], np.sqrt(meanCMid[5]), np.divide(meanCMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCMid[7], yerr = np.minimum(meanCMidRange[7], np.sqrt(meanCMid[7]), np.divide(meanCMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_400_b.png")
plt.clf()

plt.errorbar(ldaset, meanCLarge[0], yerr = np.minimum(meanCLargeRange[0], np.sqrt(meanCLarge[0]), np.divide(meanCLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCLarge[1], yerr = np.minimum(meanCLargeRange[1], np.sqrt(meanCLarge[1]), np.divide(meanCLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCLarge[2], yerr = np.minimum(meanCLargeRange[2], np.sqrt(meanCLarge[2]), np.divide(meanCLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCLarge[6], yerr = np.minimum(meanCLargeRange[6], np.sqrt(meanCLarge[6]), np.divide(meanCLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_800_a.png")
plt.clf()

plt.errorbar(ldaset, meanCLarge[3], yerr = np.minimum(meanCLargeRange[3], np.sqrt(meanCLarge[3]), np.divide(meanCLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCLarge[4], yerr = np.minimum(meanCLargeRange[4], np.sqrt(meanCLarge[4]), np.divide(meanCLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCLarge[5], yerr = np.minimum(meanCLargeRange[5], np.sqrt(meanCLarge[5]), np.divide(meanCLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCLarge[7], yerr = np.minimum(meanCLargeRange[7], np.sqrt(meanCLarge[7]), np.divide(meanCLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_synth_C_est_800_b.png")
plt.clf()

# EXPERIMENT 4: MSE of PRIEST-KLD for best lambdas extracted from experiment 3
plt.errorbar(Cset, meanSmallBest[0], yerr = np.minimum(meanSmallBestRange[0], np.sqrt(meanSmallBest[0]), np.divide(meanSmallBest[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanSmallBest[1], yerr = np.minimum(meanSmallBestRange[1], np.sqrt(meanSmallBest[1]), np.divide(meanSmallBest[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanSmallBest[2], yerr = np.minimum(meanSmallBestRange[2], np.sqrt(meanSmallBest[2]), np.divide(meanSmallBest[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanSmallBest[6], yerr = np.minimum(meanSmallBestRange[6], np.sqrt(meanSmallBest[6]), np.divide(meanSmallBest[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_40_a.png")
plt.clf()

plt.errorbar(Cset, meanSmallBest[3], yerr = np.minimum(meanSmallBestRange[3], np.sqrt(meanSmallBest[3]), np.divide(meanSmallBest[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanSmallBest[4], yerr = np.minimum(meanSmallBestRange[4], np.sqrt(meanSmallBest[4]), np.divide(meanSmallBest[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanSmallBest[5], yerr = np.minimum(meanSmallBestRange[5], np.sqrt(meanSmallBest[5]), np.divide(meanSmallBest[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanSmallBest[7], yerr = np.minimum(meanSmallBestRange[7], np.sqrt(meanSmallBest[7]), np.divide(meanSmallBest[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_40_b.png")
plt.clf()

plt.errorbar(Cset, meanDefBest[0], yerr = np.minimum(meanDefBestRange[0], np.sqrt(meanDefBest[0]), np.divide(meanDefBest[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanDefBest[1], yerr = np.minimum(meanDefBestRange[1], np.sqrt(meanDefBest[1]), np.divide(meanDefBest[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanDefBest[2], yerr = np.minimum(meanDefBestRange[2], np.sqrt(meanDefBest[2]), np.divide(meanDefBest[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanDefBest[6], yerr = np.minimum(meanDefBestRange[6], np.sqrt(meanDefBest[6]), np.divide(meanDefBest[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_200_a.png")
plt.clf()

plt.errorbar(Cset, meanDefBest[3], yerr = np.minimum(meanDefBestRange[3], np.sqrt(meanDefBest[3]), np.divide(meanDefBest[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanDefBest[4], yerr = np.minimum(meanDefBestRange[4], np.sqrt(meanDefBest[4]), np.divide(meanDefBest[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanDefBest[5], yerr = np.minimum(meanDefBestRange[5], np.sqrt(meanDefBest[5]), np.divide(meanDefBest[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanDefBest[7], yerr = np.minimum(meanDefBestRange[7], np.sqrt(meanDefBest[7]), np.divide(meanDefBest[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_200_b.png")
plt.clf()

plt.errorbar(Cset, meanMidBest[0], yerr = np.minimum(meanMidBestRange[0], np.sqrt(meanMidBest[0]), np.divide(meanMidBest[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanMidBest[1], yerr = np.minimum(meanMidBestRange[1], np.sqrt(meanMidBest[1]), np.divide(meanMidBest[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanMidBest[2], yerr = np.minimum(meanMidBestRange[2], np.sqrt(meanMidBest[2]), np.divide(meanMidBest[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanMidBest[6], yerr = np.minimum(meanMidBestRange[6], np.sqrt(meanMidBest[6]), np.divide(meanMidBest[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_400_a.png")
plt.clf()

plt.errorbar(Cset, meanMidBest[3], yerr = np.minimum(meanMidBestRange[3], np.sqrt(meanMidBest[3]), np.divide(meanMidBest[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanMidBest[4], yerr = np.minimum(meanMidBestRange[4], np.sqrt(meanMidBest[4]), np.divide(meanMidBest[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanMidBest[5], yerr = np.minimum(meanMidBestRange[5], np.sqrt(meanMidBest[5]), np.divide(meanMidBest[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanMidBest[7], yerr = np.minimum(meanMidBestRange[7], np.sqrt(meanMidBest[7]), np.divide(meanMidBest[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_400_b.png")
plt.clf()

plt.errorbar(Cset, meanLargeBest[0], yerr = np.minimum(meanLargeBestRange[0], np.sqrt(meanLargeBest[0]), np.divide(meanLargeBest[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanLargeBest[1], yerr = np.minimum(meanLargeBestRange[1], np.sqrt(meanLargeBest[1]), np.divide(meanLargeBest[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanLargeBest[2], yerr = np.minimum(meanLargeBestRange[2], np.sqrt(meanLargeBest[2]), np.divide(meanLargeBest[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanLargeBest[6], yerr = np.minimum(meanLargeBestRange[6], np.sqrt(meanLargeBest[6]), np.divide(meanLargeBest[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_800_a.png")
plt.clf()

plt.errorbar(Cset, meanLargeBest[3], yerr = np.minimum(meanLargeBestRange[3], np.sqrt(meanLargeBest[3]), np.divide(meanLargeBest[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanLargeBest[4], yerr = np.minimum(meanLargeBestRange[4], np.sqrt(meanLargeBest[4]), np.divide(meanLargeBest[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanLargeBest[5], yerr = np.minimum(meanLargeBestRange[5], np.sqrt(meanLargeBest[5]), np.divide(meanLargeBest[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanLargeBest[7], yerr = np.minimum(meanLargeBestRange[7], np.sqrt(meanLargeBest[7]), np.divide(meanLargeBest[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Number of clients " + "$\mathit{n}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_synth_C_best_800_b.png")
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