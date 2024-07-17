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

# lists of the values of C and lambda, as well as the trials that will be explored
Cset = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
          1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
trialset = ["Dist_small", "TAgg_small", "Trusted_small", "Dist_large", "TAgg_large", "Trusted_large", "no_privacy_small", "no_privacy_large"]
CS = len(Cset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanEst = np.zeros((TS, CS))
meanLdaOpt = np.zeros((TS, CS))
meanEstZero = np.zeros((TS, CS))
meanEstOne = np.zeros((TS, CS))
meanPerc = np.zeros((TS, CS))
meanCSmall = np.zeros((TS, LS))
meanCDef = np.zeros((TS, LS))
meanCMid = np.zeros((TS, LS))
meanCLarge = np.zeros((TS, LS))

meanEstRange = np.zeros((TS, CS))
meanLdaOptRange = np.zeros((TS, CS))
meanEstZeroRange = np.zeros((TS, CS))
meanEstOneRange = np.zeros((TS, CS))
meanPercRange = np.zeros((TS, CS))
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
LARGE_INDEX = 10

for trial in range(8):
    ordfile = open(f"synth_C_{trialset[trial]}.txt", "w", encoding = 'utf-8')

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
        tempMeanLdaOpt = np.zeros(RS)
        tempMeanEstZero = np.zeros(RS)
        tempMeanEstOne = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        tempMeanCSmall = np.zeros((LS, RS))
        tempMeanCDef = np.zeros((LS, RS))
        tempMeanCMid = np.zeros((LS, RS))
        tempMeanCLarge = np.zeros((LS, RS))
        
        for rep in range(RS):
            print(f"C = {C}, repeat = {rep + 1}...")
        
            # initialising seed for random sampling
            torch.manual_seed(SEED_FREQ)

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
            
                    # C = 40 (small)
                    if C_COUNT == 0:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCSmall[l, rep] = (tempMeanCSmall[l, rep] + meanSmallNoise - groundTruth)**2

                    # C = 200 (def)
                    if C_COUNT == 4:
                        meanDefNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCDef[l, rep] = (tempMeanCDef[l, rep] + meanDefNoise - groundTruth)**2

                    # C = 400 (mid)
                    if C_COUNT == 7:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCMid[l, rep] = (tempMeanCMid[l, rep] + meanMidNoise - groundTruth)**2

                    # C = 620 (large)
                    if C_COUNT == 10:
                        meanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanCLarge[l, rep] = (tempMeanCLarge[l, rep] + meanLargeNoise - groundTruth)**2

            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEst[rep] = (tempMeanEst[rep] - groundTruth)**2
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] - groundTruth)**2
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] - groundTruth)**2

                for l in range(LS):
            
                    # C = 40 (small)
                    if C_COUNT == 0:
                        tempMeanCSmall[l, rep] = (tempMeanCSmall[l, rep] - groundTruth)**2

                    # C = 200 (def)
                    if C_COUNT == 4:
                        tempMeanCDef[l, rep] = (tempMeanCDef[l, rep] - groundTruth)**2

                    # C = 400 (mid)
                    if C_COUNT == 7:
                        tempMeanCMid[l, rep] = (tempMeanCMid[l, rep] - groundTruth)**2

                    # C = 620 (large)
                    if C_COUNT == 10:
                        tempMeanCLarge[l, rep] = (tempMeanCLarge[l, rep] - groundTruth)**2

            # compute % of noise vs ground truth
            if trial % 3 == 0 and trial != 6:
                tempMeanPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise) + groundTruth))))*100
            if trial % 3 == 1 and trial != 7:
                tempMeanPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + groundTruth))*100
            if trial % 3 == 2:
                tempMeanPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise + groundTruth))))*100

            SEED_FREQ = SEED_FREQ + 1
        
        # compute mean of repeats
        meanEst[trial, C_COUNT] = np.mean(tempMeanEst)
        meanLdaOpt[trial, C_COUNT] = np.mean(tempMeanLdaOpt)
        meanEstZero[trial, C_COUNT] = np.mean(tempMeanEstZero)
        meanEstOne[trial, C_COUNT] = np.mean(tempMeanEstOne)
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
        meanEstRange[trial, C_COUNT] = np.std(tempMeanEst)
        meanLdaOptRange[trial, C_COUNT] = np.std(tempMeanLdaOpt)
        meanEstZeroRange[trial, C_COUNT] = np.std(tempMeanEstZero)
        meanEstOneRange[trial, C_COUNT] = np.std(tempMeanEstOne)
        meanPercRange[trial, C_COUNT] = np.std(tempMeanPerc)

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
            ordfile.write(f"SYNTHETIC Ordered: C = {C}\n")
        else:
            ordfile.write(f"\nC = {C}\n")

        ordfile.write(f"\nMean Error: {round(meanEst[trial, C_COUNT], 2)}\n")
        ordfile.write(f"Optimal Lambda: {round(meanLdaOpt[trial, C_COUNT], 2)}\n")
        ordfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        ordfile.write(f"Noise: {np.round(meanPerc[trial, C_COUNT], 2)}%\n")

        C_COUNT = C_COUNT + 1

# plot error of PRIEST-KLD for each C (small KLD)
plt.errorbar(Cset, meanEst[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEst[0]), np.divide(meanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEst[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEst[1]), np.divide(meanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEst[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEst[2]), np.divide(meanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEst[6], yerr = np.minimum(meanEstRange[6], np.sqrt(meanEst[6]), np.divide(meanEst[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD)
plt.errorbar(Cset, meanEst[3], yerr = np.minimum(meanEstRange[3], np.sqrt(meanEst[3]), np.divide(meanEst[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEst[4], yerr = np.minimum(meanEstRange[4], np.sqrt(meanEst[4]), np.divide(meanEst[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEst[5], yerr = np.minimum(meanEstRange[5], np.sqrt(meanEst[5]), np.divide(meanEst[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEst[7], yerr = np.minimum(meanEstRange[7], np.sqrt(meanEst[7]), np.divide(meanEst[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 0 for each C (small KLD)
plt.errorbar(Cset, meanEstZero[0], yerr = np.minimum(meanEstZeroRange[0], np.sqrt(meanEstZero[0]), np.divide(meanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstZero[1], yerr = np.minimum(meanEstZeroRange[1], np.sqrt(meanEstZero[1]), np.divide(meanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstZero[2], yerr = np.minimum(meanEstZeroRange[2], np.sqrt(meanEstZero[2]), np.divide(meanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstZero[6], yerr = np.minimum(meanEstZeroRange[6], np.sqrt(meanEstZero[6]), np.divide(meanEstZero[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_lda_zero.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 0 for each C (large KLD)
plt.errorbar(Cset, meanEstZero[3], yerr = np.minimum(meanEstZeroRange[3], np.sqrt(meanEstZero[3]), np.divide(meanEstZero[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstZero[4], yerr = np.minimum(meanEstZeroRange[4], np.sqrt(meanEstZero[4]), np.divide(meanEstZero[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstZero[5], yerr = np.minimum(meanEstZeroRange[5], np.sqrt(meanEstZero[5]), np.divide(meanEstZero[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstZero[7], yerr = np.minimum(meanEstZeroRange[7], np.sqrt(meanEstZero[7]), np.divide(meanEstZero[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_lda_zero.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 1 for each C (small KLD)
plt.errorbar(Cset, meanEstOne[0], yerr = np.minimum(meanEstOneRange[0], np.sqrt(meanEstOne[0]), np.divide(meanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstOne[1], yerr = np.minimum(meanEstOneRange[1], np.sqrt(meanEstOne[1]), np.divide(meanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstOne[2], yerr = np.minimum(meanEstOneRange[2], np.sqrt(meanEstOne[2]), np.divide(meanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstOne[6], yerr = np.minimum(meanEstOneRange[6], np.sqrt(meanEstOne[6]), np.divide(meanEstOne[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 1 for each C (large KLD)
plt.errorbar(Cset, meanEstOne[3], yerr = np.minimum(meanEstOneRange[3], np.sqrt(meanEstOne[3]), np.divide(meanEstOne[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanEstOne[4], yerr = np.minimum(meanEstOneRange[4], np.sqrt(meanEstOne[4]), np.divide(meanEstOne[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanEstOne[5], yerr = np.minimum(meanEstOneRange[5], np.sqrt(meanEstOne[5]), np.divide(meanEstOne[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanEstOne[7], yerr = np.minimum(meanEstOneRange[7], np.sqrt(meanEstOne[7]), np.divide(meanEstOne[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when C = 40 (small KLD)
plt.errorbar(ldaset, meanCSmall[0], yerr = np.minimum(meanCSmallRange[0], np.sqrt(meanCSmall[0]), np.divide(meanCSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCSmall[1], yerr = np.minimum(meanCSmallRange[1], np.sqrt(meanCSmall[1]), np.divide(meanCSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCSmall[2], yerr = np.minimum(meanCSmallRange[2], np.sqrt(meanCSmall[2]), np.divide(meanCSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCSmall[6], yerr = np.minimum(meanCSmallRange[6], np.sqrt(meanCSmall[6]), np.divide(meanCSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = 40 (large KLD)
plt.errorbar(ldaset, meanCSmall[3], yerr = np.minimum(meanCSmallRange[3], np.sqrt(meanCSmall[3]), np.divide(meanCSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCSmall[4], yerr = np.minimum(meanCSmallRange[4], np.sqrt(meanCSmall[4]), np.divide(meanCSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCSmall[5], yerr = np.minimum(meanCSmallRange[5], np.sqrt(meanCSmall[5]), np.divide(meanCSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCSmall[7], yerr = np.minimum(meanCSmallRange[7], np.sqrt(meanCSmall[7]), np.divide(meanCSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = 200 (small KLD)
plt.errorbar(ldaset, meanCDef[0], yerr = np.minimum(meanCDefRange[0], np.sqrt(meanCDef[0]), np.divide(meanCDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCDef[1], yerr = np.minimum(meanCDefRange[1], np.sqrt(meanCDef[1]), np.divide(meanCDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCDef[2], yerr = np.minimum(meanCDefRange[2], np.sqrt(meanCDef[2]), np.divide(meanCDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCDef[6], yerr = np.minimum(meanCDefRange[6], np.sqrt(meanCDef[6]), np.divide(meanCDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = 200 (large KLD)
plt.errorbar(ldaset, meanCDef[3], yerr = np.minimum(meanCDefRange[3], np.sqrt(meanCDef[3]), np.divide(meanCDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCDef[4], yerr = np.minimum(meanCDefRange[4], np.sqrt(meanCDef[4]), np.divide(meanCDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCDef[5], yerr = np.minimum(meanCDefRange[5], np.sqrt(meanCDef[5]), np.divide(meanCDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCDef[7], yerr = np.minimum(meanCDefRange[7], np.sqrt(meanCDef[7]), np.divide(meanCDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = 400 (small KLD)
plt.errorbar(ldaset, meanCMid[0], yerr = np.minimum(meanCMidRange[0], np.sqrt(meanCMid[0]), np.divide(meanCMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCMid[1], yerr = np.minimum(meanCMidRange[1], np.sqrt(meanCMid[1]), np.divide(meanCMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCMid[2], yerr = np.minimum(meanCMidRange[2], np.sqrt(meanCMid[2]), np.divide(meanCMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCMid[6], yerr = np.minimum(meanCMidRange[6], np.sqrt(meanCMid[6]), np.divide(meanCMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = 400 (large KLD)
plt.errorbar(ldaset, meanCMid[3], yerr = np.minimum(meanCMidRange[3], np.sqrt(meanCMid[3]), np.divide(meanCMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCMid[4], yerr = np.minimum(meanCMidRange[4], np.sqrt(meanCMid[4]), np.divide(meanCMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCMid[5], yerr = np.minimum(meanCMidRange[5], np.sqrt(meanCMid[5]), np.divide(meanCMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCMid[7], yerr = np.minimum(meanCMidRange[7], np.sqrt(meanCMid[7]), np.divide(meanCMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = 620 (small KLD)
plt.errorbar(ldaset, meanCLarge[0], yerr = np.minimum(meanCLargeRange[0], np.sqrt(meanCLarge[0]), np.divide(meanCLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCLarge[1], yerr = np.minimum(meanCLargeRange[1], np.sqrt(meanCLarge[1]), np.divide(meanCLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCLarge[2], yerr = np.minimum(meanCLargeRange[2], np.sqrt(meanCLarge[2]), np.divide(meanCLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCLarge[6], yerr = np.minimum(meanCLargeRange[6], np.sqrt(meanCLarge[6]), np.divide(meanCLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = 620 (large KLD)
plt.errorbar(ldaset, meanCLarge[3], yerr = np.minimum(meanCLargeRange[3], np.sqrt(meanCLarge[3]), np.divide(meanCLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanCLarge[4], yerr = np.minimum(meanCLargeRange[4], np.sqrt(meanCLarge[4]), np.divide(meanCLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanCLarge[5], yerr = np.minimum(meanCLargeRange[5], np.sqrt(meanCLarge[5]), np.divide(meanCLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanCLarge[7], yerr = np.minimum(meanCLargeRange[7], np.sqrt(meanCLarge[7]), np.divide(meanCLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_C_large.png")
plt.clf()

# plot % of noise vs ground truth for each C (small KLD)
plt.errorbar(Cset, meanPerc[0], yerr = np.minimum(meanPercRange[0], np.sqrt(meanPerc[0]), np.divide(meanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanPerc[1], yerr = np.minimum(meanPercRange[1], np.sqrt(meanPerc[1]), np.divide(meanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanPerc[2], yerr = np.minimum(meanPercRange[2], np.sqrt(meanPerc[2]), np.divide(meanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanPerc[6], yerr = np.minimum(meanPercRange[6], np.sqrt(meanPerc[6]), np.divide(meanPerc[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.savefig("Synth_C_perc_small.png")
plt.clf()

# plot % of noise vs ground truth for each C (large KLD)
plt.errorbar(Cset, meanPerc[3], yerr = np.minimum(meanPercRange[3], np.sqrt(meanPerc[3]), np.divide(meanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, meanPerc[4], yerr = np.minimum(meanPercRange[4], np.sqrt(meanPerc[4]), np.divide(meanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, meanPerc[5], yerr = np.minimum(meanPercRange[5], np.sqrt(meanPerc[5]), np.divide(meanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, meanPerc[7], yerr = np.minimum(meanPercRange[7], np.sqrt(meanPerc[7]), np.divide(meanPerc[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.savefig("Synth_C_perc_large.png")
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