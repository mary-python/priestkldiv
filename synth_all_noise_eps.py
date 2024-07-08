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
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist_small", "TAgg_small", "Trusted_small", "Dist_large", "TAgg_large", "Trusted_large", "no_privacy_small", "no_privacy_large"]
ES = len(epsset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates (random +/-)
rMeanEst = np.zeros((TS, ES))
rLdaOpt = np.zeros((TS, ES))
rMeanEstZero = np.zeros((TS, ES))
rMeanEstOne = np.zeros((TS, ES))
rMeanPerc = np.zeros((TS, ES))
rMeanEpsSmall = np.zeros((TS, LS))
rMeanEpsDef = np.zeros((TS, LS))
rMeanEpsMid = np.zeros((TS, LS))
rMeanEpsLarge = np.zeros((TS, LS))

rMeanEstRange = np.zeros((TS, ES))
rLdaOptRange = np.zeros((TS, ES))
rMeanEstZeroRange = np.zeros((TS, ES))
rMeanEstOneRange = np.zeros((TS, ES))
rMeanPercRange = np.zeros((TS, ES))
rMeanEpsSmallRange = np.zeros((TS, LS))
rMeanEpsDefRange = np.zeros((TS, LS))
rMeanEpsMidRange = np.zeros((TS, LS))
rMeanEpsLargeRange = np.zeros((TS, LS))

# to store statistics related to mean estimates (ordered)
oMeanEst = np.zeros((TS, ES))
oLdaOpt = np.zeros((TS, ES))
oMeanEstZero = np.zeros((TS, ES))
oMeanEstOne = np.zeros((TS, ES))
oMeanPerc = np.zeros((TS, ES))
oMeanEpsSmall = np.zeros((TS, LS))
oMeanEpsDef = np.zeros((TS, LS))
oMeanEpsMid = np.zeros((TS, LS))
oMeanEpsLarge = np.zeros((TS, LS))

oMeanEstRange = np.zeros((TS, ES))
oLdaOptRange = np.zeros((TS, ES))
oMeanEstZeroRange = np.zeros((TS, ES))
oMeanEstOneRange = np.zeros((TS, ES))
oMeanPercRange = np.zeros((TS, ES))
oMeanEpsSmallRange = np.zeros((TS, LS))
oMeanEpsDefRange = np.zeros((TS, LS))
oMeanEpsMidRange = np.zeros((TS, LS))
oMeanEpsLargeRange = np.zeros((TS, LS))

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
    randfile = open(f"synth_eps_{trialset[trial]}_rand.txt", "w", encoding = 'utf-8')
    ordfile = open(f"synth_eps_{trialset[trial]}_ord.txt", "w", encoding = 'utf-8')

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

    # order the pre-processed sample and separate into + and - values
    qOrderedRound = torch.sort(qRound)
    qNegativeRound = qOrderedRound[0][0:249151]
    qPositiveRound = qOrderedRound[0][249151:499947]
    
    if trial < 6:
        b1 = 1 + log(2)
        b2 = 2*((log(1.25))/DTA)*b1

    EPS_COUNT = 0

    for eps in epsset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        rTempMeanEst = np.zeros(RS)
        rTempLdaOpt = np.zeros(RS)
        rTempMeanEstZero = np.zeros(RS)
        rTempMeanEstOne = np.zeros(RS)
        rTempMeanPerc = np.zeros(RS)
        rTempMeanEpsSmall = np.zeros((LS, RS))
        rTempMeanEpsDef = np.zeros((LS, RS))
        rTempMeanEpsMid = np.zeros((LS, RS))
        rTempMeanEpsLarge = np.zeros((LS, RS))

        oTempMeanEst = np.zeros(RS)
        oTempLdaOpt = np.zeros(RS)
        oTempMeanEstZero = np.zeros(RS)
        oTempMeanEstOne = np.zeros(RS)
        oTempMeanPerc = np.zeros(RS)
        oTempMeanEpsSmall = np.zeros((LS, RS))
        oTempMeanEpsDef = np.zeros((LS, RS))
        oTempMeanEpsMid = np.zeros((LS, RS))
        oTempMeanEpsLarge = np.zeros((LS, RS))

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

            # numpy arrays
            rEst = np.zeros((LS, C))
            oEst = np.zeros((LS, C))
            rStartNoise = []
            oStartNoise = []

            for j in range(C):

                # "RANDOM +/-": even clients get positive values, odd clients get negative values
                if (j % 2) == 0:
                    indices = torch.randperm(len(qPositiveRound))[:N]
                    qClientSamp = qPositiveRound[indices]
                else:
                    indices = torch.randperm(len(qNegativeRound))[:N]
                    qClientSamp = qNegativeRound[indices]

                # "ORDERED": each client gets N points in order from ordered pre-processed sample
                qOrdClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

                # compute ratio between unknown and known distributions
                rLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
                oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)

                # "Dist" (each client adds Gaussian noise term)
                if trial % 3 == 0 and trial != 6:
                    rStartSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                    oStartSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                    rStartNoise.append(rStartSample)
                    oStartNoise.append(oStartSample)
                    rLogr = rLogr + rStartSample
                    oLogr = oLogr + rStartSample

                LDA_COUNT = 0

                # explore lambdas in a range
                for lda in ldaset:

                    # compute k3 estimator
                    rRangeEst = lda * (np.exp(rLogr) - 1) - rLogr
                    oRangeEst = lda * (np.exp(oLogr) - 1) - oLogr

                    # share PRIEST-KLD with server
                    rEst[LDA_COUNT, j] = rRangeEst.mean()
                    oEst[LDA_COUNT, j] = oRangeEst.mean()
                    LDA_COUNT = LDA_COUNT + 1

            # compute mean of PRIEST-KLD across clients
            rMeanLda = np.mean(rEst, axis = 1)
            oMeanLda = np.mean(oEst, axis = 1)

            rMeanLdaNoise = np.zeros(LS)
            oMeanLdaNoise = np.zeros(LS)
        
            for l in range(LS):

                # "TAgg" (intermediate server adds Gaussian noise term)
                if trial % 3 == 1 and trial != 7:
                    rMeanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    oMeanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    rMeanLda[l] = rMeanLda[l] + rMeanLdaNoise[l]
                    oMeanLda[l] = oMeanLda[l] + oMeanLdaNoise[l]
            
                # mean across lambdas for eps = 0.05 (small)
                if EPS_COUNT == SMALL_INDEX:
                    rTempMeanEpsSmall[l, rep] = rMeanLda[l]
                    oTempMeanEpsSmall[l, rep] = oMeanLda[l]

                # eps = 0.5 (default)
                if EPS_COUNT == DEF_INDEX:
                    rTempMeanEpsDef[l, rep] = rMeanLda[l]
                    oTempMeanEpsDef[l, rep] = oMeanLda[l]

                # eps = 1.5 (mid)
                if EPS_COUNT == MID_INDEX:
                    rTempMeanEpsMid[l, rep] = rMeanLda[l]
                    oTempMeanEpsMid[l, rep] = oMeanLda[l]

                # eps = 3 (large)
                if EPS_COUNT == LARGE_INDEX:
                    rTempMeanEpsLarge[l, rep] = rMeanLda[l]
                    oTempMeanEpsLarge[l, rep] = oMeanLda[l]

            # find lambda that produces minimum error
            rLdaIndex = np.argmin(rMeanLda)
            oLdaIndex = np.argmin(oMeanLda)

            rMinMeanError = rMeanLda[rLdaIndex]
            oMinMeanError = oMeanLda[oLdaIndex]

            # mean across clients for optimum lambda
            rTempMeanEst[rep] = rMinMeanError
            oTempMeanEst[rep] = oMinMeanError

            # optimum lambda
            rTempLdaOpt[rep] = rLdaIndex * ldaStep
            oTempLdaOpt[rep] = oLdaIndex * ldaStep

            # lambda = 0
            rTempMeanEstZero[rep] = rMeanLda[0]
            oTempMeanEstZero[rep] = oMeanLda[0]

            # lambda = 1
            rTempMeanEstOne[rep] = rMeanLda[LS-1]
            oTempMeanEstOne[rep] = oMeanLda[LS-1]

            # "Trusted" (server adds Laplace noise term to final result)
            if trial % 3 == 2:
                rMeanNoise = lapNoise.sample(sample_shape = (1,))
                oMeanNoise = lapNoise.sample(sample_shape = (1,))
                rMeanZeroNoise = lapNoise.sample(sample_shape = (1,))
                oMeanZeroNoise = lapNoise.sample(sample_shape = (1,))
                rMeanOneNoise = lapNoise.sample(sample_shape = (1,))
                oMeanOneNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                rTempMeanEst[rep] = (rTempMeanEst[rep] + rMeanNoise - groundTruth)**2
                oTempMeanEst[rep] = (oTempMeanEst[rep] + oMeanNoise - groundTruth)**2

                # lambda = 0
                rTempMeanEstZero[rep] = (rTempMeanEstZero[rep] + rMeanZeroNoise - groundTruth)**2
                oTempMeanEstZero[rep] = (oTempMeanEstZero[rep] + oMeanZeroNoise - groundTruth)**2

                # lambda = 1
                rTempMeanEstOne[rep] = (rTempMeanEstOne[rep] + rMeanOneNoise - groundTruth)**2
                oTempMeanEstOne[rep] = (oTempMeanEstOne[rep] + oMeanOneNoise - groundTruth)**2

                for l in range(LS):
            
                    # eps = 0.05 (small)
                    if EPS_COUNT == 0:
                        rMeanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanEpsSmall[l, rep] = (rTempMeanEpsSmall[l, rep] + rMeanSmallNoise - groundTruth)**2
                        oTempMeanEpsSmall[l, rep] = (oTempMeanEpsSmall[l, rep] + oMeanSmallNoise - groundTruth)**2

                    # eps = 0.5 (def)
                    if EPS_COUNT == 3:
                        rMeanDefNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanDefNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanEpsDef[l, rep] = (rTempMeanEpsDef[l, rep] + rMeanDefNoise - groundTruth)**2
                        oTempMeanEpsDef[l, rep] = (oTempMeanEpsDef[l, rep] + oMeanDefNoise - groundTruth)**2

                    # eps = 1.5 (mid)
                    if EPS_COUNT == 6:
                        rMeanMidNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanMidNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanEpsMid[l, rep] = (rTempMeanEpsMid[l, rep] + rMeanMidNoise - groundTruth)**2
                        oTempMeanEpsMid[l, rep] = (oTempMeanEpsMid[l, rep] + oMeanMidNoise - groundTruth)**2

                    # eps = 3 (large)
                    if EPS_COUNT == 10:
                        rMeanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanEpsLarge[l, rep] = (rTempMeanEpsLarge[l, rep] + rMeanLargeNoise - groundTruth)**2
                        oTempMeanEpsLarge[l, rep] = (oTempMeanEpsLarge[l, rep] + oMeanLargeNoise - groundTruth)**2

            # clients or intermediate server already added Gaussian noise term
            else:
                rTempMeanEst[rep] = (rTempMeanEst[rep] - groundTruth)**2
                oTempMeanEst[rep] = (oTempMeanEst[rep] - groundTruth)**2

                # lambda = 0
                rTempMeanEstZero[rep] = (rTempMeanEstZero[rep] - groundTruth)**2
                oTempMeanEstZero[rep] = (oTempMeanEstZero[rep] - groundTruth)**2

                # lambda = 1
                rTempMeanEstOne[rep] = (rTempMeanEstOne[rep] - groundTruth)**2
                oTempMeanEstOne[rep] = (oTempMeanEstOne[rep] - groundTruth)**2

                for l in range(LS):
            
                    # eps = 0.05 (small)
                    if EPS_COUNT == 0:
                        rTempMeanEpsSmall[l, rep] = (rTempMeanEpsSmall[l, rep] - groundTruth)**2
                        oTempMeanEpsSmall[l, rep] = (oTempMeanEpsSmall[l, rep] - groundTruth)**2

                    # eps = 0.5 (def)
                    if EPS_COUNT == 3:
                        rTempMeanEpsDef[l, rep] = (rTempMeanEpsDef[l, rep] - groundTruth)**2
                        oTempMeanEpsDef[l, rep] = (oTempMeanEpsDef[l, rep] - groundTruth)**2

                    # eps = 1.5 (mid)
                    if EPS_COUNT == 6:
                        rTempMeanEpsMid[l, rep] = (rTempMeanEpsMid[l, rep] - groundTruth)**2
                        oTempMeanEpsMid[l, rep] = (oTempMeanEpsMid[l, rep] - groundTruth)**2

                    # eps = 3 (large)
                    if EPS_COUNT == 10:
                        rTempMeanEpsLarge[l, rep] = (rTempMeanEpsLarge[l, rep] - groundTruth)**2
                        oTempMeanEpsLarge[l, rep] = (oTempMeanEpsLarge[l, rep] - groundTruth)**2

            # compute % of noise vs ground truth
            if trial % 3 == 0 and trial != 6:
                rTempMeanPerc[rep] = float(abs(np.array(sum(rStartNoise)) / (np.array(sum(rStartNoise) + groundTruth))))*100
            if trial % 3 == 1 and trial != 7:
                rTempMeanPerc[rep] = abs((np.sum(rMeanLdaNoise)) / (np.sum(rMeanLdaNoise) + groundTruth))*100
            if trial % 3 == 2:
                rTempMeanPerc[rep] = float(abs(np.array(rMeanNoise) / (np.array(rMeanNoise + groundTruth))))*100

            if trial % 3 == 0 and trial != 6:
                oTempMeanPerc[rep] = float(abs(np.array(sum(oStartNoise)) / (np.array(sum(oStartNoise) + groundTruth))))*100
            if trial % 3 == 1 and trial != 7:
                oTempMeanPerc[rep] = abs((np.sum(oMeanLdaNoise)) / (np.sum(oMeanLdaNoise) + groundTruth))*100
            if trial % 3 == 2:
                oTempMeanPerc[rep] = float(abs(np.array(oMeanNoise) / (np.array(oMeanNoise + groundTruth))))*100
        
            SEED_FREQ = SEED_FREQ + 1

        # compute mean of repeats
        rMeanEst[trial, EPS_COUNT] = np.mean(rTempMeanEst)
        rLdaOpt[trial, EPS_COUNT] = np.mean(rTempLdaOpt)
        rMeanEstZero[trial, EPS_COUNT] = np.mean(rTempMeanEstZero)
        rMeanEstOne[trial, EPS_COUNT] = np.mean(rTempMeanEstOne)
        rMeanPerc[trial, EPS_COUNT] = np.mean(rTempMeanPerc)

        for l in range(LS):
            if EPS_COUNT == SMALL_INDEX:
                rMeanEpsSmall[trial, l] = np.mean(rTempMeanEpsSmall[l])
            if EPS_COUNT == DEF_INDEX:
                rMeanEpsDef[trial, l] = np.mean(rTempMeanEpsDef[l])
            if EPS_COUNT == MID_INDEX:
                rMeanEpsMid[trial, l] = np.mean(rTempMeanEpsMid[l])
            if EPS_COUNT == LARGE_INDEX:
                rMeanEpsLarge[trial, l] = np.mean(rTempMeanEpsLarge[l])

        oMeanEst[trial, EPS_COUNT] = np.mean(oTempMeanEst)
        oLdaOpt[trial, EPS_COUNT] = np.mean(oTempLdaOpt)
        oMeanEstZero[trial, EPS_COUNT] = np.mean(oTempMeanEstZero)
        oMeanEstOne[trial, EPS_COUNT] = np.mean(oTempMeanEstOne)
        oMeanPerc[trial, EPS_COUNT] = np.mean(oTempMeanPerc)

        for l in range(LS):
            if EPS_COUNT == SMALL_INDEX:
                oMeanEpsSmall[trial, l] = np.mean(oTempMeanEpsSmall[l])
            if EPS_COUNT == DEF_INDEX:
                oMeanEpsDef[trial, l] = np.mean(oTempMeanEpsDef[l])
            if EPS_COUNT == MID_INDEX:
                oMeanEpsMid[trial, l] = np.mean(oTempMeanEpsMid[l])
            if EPS_COUNT == LARGE_INDEX:
                oMeanEpsLarge[trial, l] = np.mean(oTempMeanEpsLarge[l])
        
        # compute standard deviation of repeats
        rMeanEstRange[trial, EPS_COUNT] = np.std(rTempMeanEst)
        rLdaOptRange[trial, EPS_COUNT] = np.std(rTempLdaOpt)
        rMeanEstZeroRange[trial, EPS_COUNT] = np.std(rTempMeanEstZero)
        rMeanEstOneRange[trial, EPS_COUNT] = np.std(rTempMeanEstOne)
        rMeanPercRange[trial, EPS_COUNT] = np.std(rTempMeanPerc)

        for l in range(LS):
            if EPS_COUNT == SMALL_INDEX:
                rMeanEpsSmallRange[trial, l] = np.std(rTempMeanEpsSmall[l])
            if EPS_COUNT == DEF_INDEX:
                rMeanEpsDefRange[trial, l] = np.std(rTempMeanEpsDef[l])
            if EPS_COUNT == MID_INDEX:
                rMeanEpsMidRange[trial, l] = np.std(rTempMeanEpsMid[l])
            if EPS_COUNT == LARGE_INDEX:
                rMeanEpsLargeRange[trial, l] = np.std(rTempMeanEpsLarge[l])

        oMeanEstRange[trial, EPS_COUNT] = np.std(oTempMeanEst)
        oLdaOptRange[trial, EPS_COUNT] = np.std(oTempLdaOpt)
        oMeanEstZeroRange[trial, EPS_COUNT] = np.std(oTempMeanEstZero)
        oMeanEstOneRange[trial, EPS_COUNT] = np.std(oTempMeanEstOne)
        oMeanPercRange[trial, EPS_COUNT] = np.std(oTempMeanPerc)

        for l in range(LS):
            if EPS_COUNT == SMALL_INDEX:
                oMeanEpsSmallRange[trial, l] = np.std(oTempMeanEpsSmall[l])
            if EPS_COUNT == DEF_INDEX:
                oMeanEpsDefRange[trial, l] = np.std(oTempMeanEpsDef[l])
            if EPS_COUNT == MID_INDEX:
                oMeanEpsMidRange[trial, l] = np.std(oTempMeanEpsMid[l])
            if EPS_COUNT == LARGE_INDEX:
                oMeanEpsLargeRange[trial, l] = np.std(oTempMeanEpsLarge[l])

        # write statistics on data files
        if eps == epsset[0]:
            randfile.write(f"SYNTHETIC Random +/-: Eps = {eps}\n")
            ordfile.write(f"SYNTHETIC Ordered: Eps = {eps}\n")
        else:
            randfile.write(f"\nEps = {eps}\n")
            ordfile.write(f"\nEps = {eps}\n")

        randfile.write(f"\nMean Error: {round(rMeanEst[trial, EPS_COUNT], 2)}\n")
        randfile.write(f"Optimal Lambda: {round(rLdaOpt[trial, EPS_COUNT], 2)}\n")
        randfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        randfile.write(f"Noise: {np.round(rMeanPerc[trial, EPS_COUNT], 2)}%\n")

        ordfile.write(f"\nMean Error: {round(oMeanEst[trial, EPS_COUNT], 2)}\n")
        ordfile.write(f"Optimal Lambda: {round(oLdaOpt[trial, EPS_COUNT], 2)}\n")
        ordfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        ordfile.write(f"Noise: {np.round(oMeanPerc[trial, EPS_COUNT], 2)}%\n")

        EPS_COUNT = EPS_COUNT + 1

# plot error of PRIEST-KLD for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rMeanEst[0], yerr = np.minimum(rMeanEstRange[0], np.sqrt(rMeanEst[0]), np.divide(rMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEst[1], yerr = np.minimum(rMeanEstRange[1], np.sqrt(rMeanEst[1]), np.divide(rMeanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEst[2], yerr = np.minimum(rMeanEstRange[2], np.sqrt(rMeanEst[2]), np.divide(rMeanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oMeanEst[0], yerr = np.minimum(oMeanEstRange[0], np.sqrt(oMeanEst[0]), np.divide(oMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEst[1], yerr = np.minimum(oMeanEstRange[1], np.sqrt(oMeanEst[1]), np.divide(oMeanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEst[2], yerr = np.minimum(oMeanEstRange[2], np.sqrt(oMeanEst[2]), np.divide(oMeanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rMeanEst[3], yerr = np.minimum(rMeanEstRange[3], np.sqrt(rMeanEst[3]), np.divide(rMeanEst[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEst[4], yerr = np.minimum(rMeanEstRange[4], np.sqrt(rMeanEst[4]), np.divide(rMeanEst[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEst[5], yerr = np.minimum(rMeanEstRange[5], np.sqrt(rMeanEst[5]), np.divide(rMeanEst[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oMeanEst[3], yerr = np.minimum(oMeanEstRange[3], np.sqrt(oMeanEst[3]), np.divide(oMeanEst[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEst[4], yerr = np.minimum(oMeanEstRange[4], np.sqrt(oMeanEst[4]), np.divide(oMeanEst[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEst[5], yerr = np.minimum(oMeanEstRange[5], np.sqrt(oMeanEst[5]), np.divide(oMeanEst[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord.png")
plt.clf()

# plot optimum lambda for each epsilon (small KLD, random +/-)
plt.plot(epsset, rLdaOpt[0], yerr = np.minimum(rLdaOptRange[0], np.sqrt(rLdaOpt[0]), np.divide(rLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rLdaOpt[1], yerr = np.minimum(rLdaOptRange[1], np.sqrt(rLdaOpt[1]), np.divide(rLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rLdaOpt[2], yerr = np.minimum(rLdaOptRange[2], np.sqrt(rLdaOpt[2]), np.divide(rLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_eps_lda_opt_small_rand.png")
plt.clf()

# plot optimum lambda for each epsilon (small KLD, ordered)
plt.plot(epsset, oLdaOpt[0], yerr = np.minimum(oLdaOptRange[0], np.sqrt(oLdaOpt[0]), np.divide(oLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oLdaOpt[1], yerr = np.minimum(oLdaOptRange[1], np.sqrt(oLdaOpt[1]), np.divide(oLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oLdaOpt[2], yerr = np.minimum(oLdaOptRange[2], np.sqrt(oLdaOpt[2]), np.divide(oLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_eps_lda_opt_small_ord.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, random +/-)
plt.plot(epsset, rLdaOpt[3], yerr = np.minimum(rLdaOptRange[3], np.sqrt(rLdaOpt[3]), np.divide(rLdaOpt[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rLdaOpt[4], yerr = np.minimum(rLdaOptRange[4], np.sqrt(rLdaOpt[4]), np.divide(rLdaOpt[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rLdaOpt[5], yerr = np.minimum(rLdaOptRange[5], np.sqrt(rLdaOpt[5]), np.divide(rLdaOpt[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_eps_lda_opt_large_rand.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, ordered)
plt.plot(epsset, oLdaOpt[3], yerr = np.minimum(oLdaOptRange[3], np.sqrt(oLdaOpt[3]), np.divide(oLdaOpt[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oLdaOpt[4], yerr = np.minimum(oLdaOptRange[4], np.sqrt(oLdaOpt[4]), np.divide(oLdaOpt[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oLdaOpt[5], yerr = np.minimum(oLdaOptRange[5], np.sqrt(oLdaOpt[5]), np.divide(oLdaOpt[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_eps_lda_opt_large_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rMeanEstZero[0], yerr = np.minimum(rMeanEstZeroRange[0], np.sqrt(rMeanEstZero[0]), np.divide(rMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstZero[1], yerr = np.minimum(rMeanEstZeroRange[1], np.sqrt(rMeanEstZero[1]), np.divide(rMeanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstZero[2], yerr = np.minimum(rMeanEstZeroRange[2], np.sqrt(rMeanEstZero[2]), np.divide(rMeanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oMeanEstZero[0], yerr = np.minimum(oMeanEstZeroRange[0], np.sqrt(oMeanEstZero[0]), np.divide(oMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstZero[1], yerr = np.minimum(oMeanEstZeroRange[1], np.sqrt(oMeanEstZero[1]), np.divide(oMeanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstZero[2], yerr = np.minimum(oMeanEstZeroRange[2], np.sqrt(oMeanEstZero[2]), np.divide(oMeanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rMeanEstZero[3], yerr = np.minimum(rMeanEstZeroRange[3], np.sqrt(rMeanEstZero[3]), np.divide(rMeanEstZero[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstZero[4], yerr = np.minimum(rMeanEstZeroRange[4], np.sqrt(rMeanEstZero[4]), np.divide(rMeanEstZero[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstZero[5], yerr = np.minimum(rMeanEstZeroRange[5], np.sqrt(rMeanEstZero[5]), np.divide(rMeanEstZero[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oMeanEstZero[3], yerr = np.minimum(oMeanEstZeroRange[3], np.sqrt(oMeanEstZero[3]), np.divide(oMeanEstZero[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstZero[4], yerr = np.minimum(oMeanEstZeroRange[4], np.sqrt(oMeanEstZero[4]), np.divide(oMeanEstZero[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstZero[5], yerr = np.minimum(oMeanEstZeroRange[5], np.sqrt(oMeanEstZero[5]), np.divide(oMeanEstZero[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rMeanEstOne[0], yerr = np.minimum(rMeanEstOneRange[0], np.sqrt(rMeanEstOne[0]), np.divide(rMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstOne[1], yerr = np.minimum(rMeanEstOneRange[1], np.sqrt(rMeanEstOne[1]), np.divide(rMeanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstOne[2], yerr = np.minimum(rMeanEstOneRange[2], np.sqrt(rMeanEstOne[2]), np.divide(rMeanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oMeanEstOne[0], yerr = np.minimum(oMeanEstOneRange[0], np.sqrt(oMeanEstOne[0]), np.divide(oMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstOne[1], yerr = np.minimum(oMeanEstOneRange[1], np.sqrt(oMeanEstOne[1]), np.divide(oMeanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstOne[2], yerr = np.minimum(oMeanEstOneRange[2], np.sqrt(oMeanEstOne[2]), np.divide(oMeanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rMeanEstOne[3], yerr = np.minimum(rMeanEstOneRange[3], np.sqrt(rMeanEstOne[3]), np.divide(rMeanEstOne[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstOne[4], yerr = np.minimum(rMeanEstOneRange[4], np.sqrt(rMeanEstOne[4]), np.divide(rMeanEstOne[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstOne[5], yerr = np.minimum(rMeanEstOneRange[5], np.sqrt(rMeanEstOne[5]), np.divide(rMeanEstOne[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oMeanEstOne[3], yerr = np.minimum(oMeanEstOneRange[3], np.sqrt(oMeanEstOne[3]), np.divide(oMeanEstOne[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstOne[4], yerr = np.minimum(oMeanEstOneRange[4], np.sqrt(oMeanEstOne[4]), np.divide(oMeanEstOne[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstOne[5], yerr = np.minimum(oMeanEstOneRange[5], np.sqrt(oMeanEstOne[5]), np.divide(oMeanEstOne[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsSmall[0], yerr = np.minimum(rMeanEpsSmallRange[0], np.sqrt(rMeanEpsSmall[0]), np.divide(rMeanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsSmall[1], yerr = np.minimum(rMeanEpsSmallRange[1], np.sqrt(rMeanEpsSmall[1]), np.divide(rMeanEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsSmall[2], yerr = np.minimum(rMeanEpsSmallRange[2], np.sqrt(rMeanEpsSmall[2]), np.divide(rMeanEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsSmall[6], yerr = np.minimum(rMeanEpsSmallRange[6], np.sqrt(rMeanEpsSmall[6]), np.divide(rMeanEpsSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsSmall[0], yerr = np.minimum(oMeanEpsSmallRange[0], np.sqrt(oMeanEpsSmall[0]), np.divide(oMeanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsSmall[1], yerr = np.minimum(oMeanEpsSmallRange[1], np.sqrt(oMeanEpsSmall[1]), np.divide(oMeanEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsSmall[2], yerr = np.minimum(oMeanEpsSmallRange[2], np.sqrt(oMeanEpsSmall[2]), np.divide(oMeanEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsSmall[6], yerr = np.minimum(oMeanEpsSmallRange[6], np.sqrt(oMeanEpsSmall[6]), np.divide(oMeanEpsSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsSmall[3], yerr = np.minimum(rMeanEpsSmallRange[3], np.sqrt(rMeanEpsSmall[3]), np.divide(rMeanEpsSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsSmall[4], yerr = np.minimum(rMeanEpsSmallRange[4], np.sqrt(rMeanEpsSmall[4]), np.divide(rMeanEpsSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsSmall[5], yerr = np.minimum(rMeanEpsSmallRange[5], np.sqrt(rMeanEpsSmall[5]), np.divide(rMeanEpsSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsSmall[7], yerr = np.minimum(rMeanEpsSmallRange[7], np.sqrt(rMeanEpsSmall[7]), np.divide(rMeanEpsSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsSmall[3], yerr = np.minimum(oMeanEpsSmallRange[3], np.sqrt(oMeanEpsSmall[3]), np.divide(oMeanEpsSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsSmall[4], yerr = np.minimum(oMeanEpsSmallRange[4], np.sqrt(oMeanEpsSmall[4]), np.divide(oMeanEpsSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsSmall[5], yerr = np.minimum(oMeanEpsSmallRange[5], np.sqrt(oMeanEpsSmall[5]), np.divide(oMeanEpsSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsSmall[7], yerr = np.minimum(oMeanEpsSmallRange[7], np.sqrt(oMeanEpsSmall[7]), np.divide(oMeanEpsSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsDef[0], yerr = np.minimum(rMeanEpsDefRange[0], np.sqrt(rMeanEpsDef[0]), np.divide(rMeanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsDef[1], yerr = np.minimum(rMeanEpsDefRange[1], np.sqrt(rMeanEpsDef[1]), np.divide(rMeanEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsDef[2], yerr = np.minimum(rMeanEpsDefRange[2], np.sqrt(rMeanEpsDef[2]), np.divide(rMeanEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsDef[6], yerr = np.minimum(rMeanEpsDefRange[6], np.sqrt(rMeanEpsDef[6]), np.divide(rMeanEpsDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsDef[0], yerr = np.minimum(oMeanEpsDefRange[0], np.sqrt(oMeanEpsDef[0]), np.divide(oMeanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsDef[1], yerr = np.minimum(oMeanEpsDefRange[1], np.sqrt(oMeanEpsDef[1]), np.divide(oMeanEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsDef[2], yerr = np.minimum(oMeanEpsDefRange[2], np.sqrt(oMeanEpsDef[2]), np.divide(oMeanEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsDef[6], yerr = np.minimum(oMeanEpsDefRange[6], np.sqrt(oMeanEpsDef[6]), np.divide(oMeanEpsDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsDef[3], yerr = np.minimum(rMeanEpsDefRange[3], np.sqrt(rMeanEpsDef[3]), np.divide(rMeanEpsDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsDef[4], yerr = np.minimum(rMeanEpsDefRange[4], np.sqrt(rMeanEpsDef[4]), np.divide(rMeanEpsDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsDef[5], yerr = np.minimum(rMeanEpsDefRange[5], np.sqrt(rMeanEpsDef[5]), np.divide(rMeanEpsDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsDef[7], yerr = np.minimum(rMeanEpsDefRange[7], np.sqrt(rMeanEpsDef[7]), np.divide(rMeanEpsDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsDef[3], yerr = np.minimum(oMeanEpsDefRange[3], np.sqrt(oMeanEpsDef[3]), np.divide(oMeanEpsDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsDef[4], yerr = np.minimum(oMeanEpsDefRange[4], np.sqrt(oMeanEpsDef[4]), np.divide(oMeanEpsDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsDef[5], yerr = np.minimum(oMeanEpsDefRange[5], np.sqrt(oMeanEpsDef[5]), np.divide(oMeanEpsDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsDef[7], yerr = np.minimum(oMeanEpsDefRange[7], np.sqrt(oMeanEpsDef[7]), np.divide(oMeanEpsDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsMid[0], yerr = np.minimum(rMeanEpsMidRange[0], np.sqrt(rMeanEpsMid[0]), np.divide(rMeanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsMid[1], yerr = np.minimum(rMeanEpsMidRange[1], np.sqrt(rMeanEpsMid[1]), np.divide(rMeanEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsMid[2], yerr = np.minimum(rMeanEpsMidRange[2], np.sqrt(rMeanEpsMid[2]), np.divide(rMeanEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsMid[6], yerr = np.minimum(rMeanEpsMidRange[6], np.sqrt(rMeanEpsMid[6]), np.divide(rMeanEpsMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsMid[0], yerr = np.minimum(oMeanEpsMidRange[0], np.sqrt(oMeanEpsMid[0]), np.divide(oMeanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsMid[1], yerr = np.minimum(oMeanEpsMidRange[1], np.sqrt(oMeanEpsMid[1]), np.divide(oMeanEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsMid[2], yerr = np.minimum(oMeanEpsMidRange[2], np.sqrt(oMeanEpsMid[2]), np.divide(oMeanEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsMid[6], yerr = np.minimum(oMeanEpsMidRange[6], np.sqrt(oMeanEpsMid[6]), np.divide(oMeanEpsMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsMid[3], yerr = np.minimum(rMeanEpsMidRange[3], np.sqrt(rMeanEpsMid[3]), np.divide(rMeanEpsMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsMid[4], yerr = np.minimum(rMeanEpsMidRange[4], np.sqrt(rMeanEpsMid[4]), np.divide(rMeanEpsMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsMid[5], yerr = np.minimum(rMeanEpsMidRange[5], np.sqrt(rMeanEpsMid[5]), np.divide(rMeanEpsMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsMid[7], yerr = np.minimum(rMeanEpsMidRange[7], np.sqrt(rMeanEpsMid[7]), np.divide(rMeanEpsMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsMid[3], yerr = np.minimum(oMeanEpsMidRange[3], np.sqrt(oMeanEpsMid[3]), np.divide(oMeanEpsMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsMid[4], yerr = np.minimum(oMeanEpsMidRange[4], np.sqrt(oMeanEpsMid[4]), np.divide(oMeanEpsMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsMid[5], yerr = np.minimum(oMeanEpsMidRange[5], np.sqrt(oMeanEpsMid[5]), np.divide(oMeanEpsMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsMid[7], yerr = np.minimum(oMeanEpsMidRange[7], np.sqrt(oMeanEpsMid[7]), np.divide(oMeanEpsMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsLarge[0], yerr = np.minimum(rMeanEpsLargeRange[0], np.sqrt(rMeanEpsLarge[0]), np.divide(rMeanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsLarge[1], yerr = np.minimum(rMeanEpsLargeRange[1], np.sqrt(rMeanEpsLarge[1]), np.divide(rMeanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsLarge[2], yerr = np.minimum(rMeanEpsLargeRange[2], np.sqrt(rMeanEpsLarge[2]), np.divide(rMeanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsLarge[6], yerr = np.minimum(rMeanEpsLargeRange[6], np.sqrt(rMeanEpsLarge[6]), np.divide(rMeanEpsLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_rand_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsLarge[0], yerr = np.minimum(oMeanEpsLargeRange[0], np.sqrt(oMeanEpsLarge[0]), np.divide(oMeanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsLarge[1], yerr = np.minimum(oMeanEpsLargeRange[1], np.sqrt(oMeanEpsLarge[1]), np.divide(oMeanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsLarge[2], yerr = np.minimum(oMeanEpsLargeRange[2], np.sqrt(oMeanEpsLarge[2]), np.divide(oMeanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsLarge[6], yerr = np.minimum(oMeanEpsLargeRange[6], np.sqrt(oMeanEpsLarge[6]), np.divide(oMeanEpsLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_small_ord_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsLarge[3], yerr = np.minimum(rMeanEpsLargeRange[3], np.sqrt(rMeanEpsLarge[3]), np.divide(rMeanEpsLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsLarge[4], yerr = np.minimum(rMeanEpsLargeRange[4], np.sqrt(rMeanEpsLarge[4]), np.divide(rMeanEpsLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsLarge[5], yerr = np.minimum(rMeanEpsLargeRange[5], np.sqrt(rMeanEpsLarge[5]), np.divide(rMeanEpsLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsLarge[7], yerr = np.minimum(rMeanEpsLargeRange[7], np.sqrt(rMeanEpsLarge[7]), np.divide(rMeanEpsLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_rand_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsLarge[3], yerr = np.minimum(oMeanEpsLargeRange[3], np.sqrt(oMeanEpsLarge[3]), np.divide(oMeanEpsLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsLarge[4], yerr = np.minimum(oMeanEpsLargeRange[4], np.sqrt(oMeanEpsLarge[4]), np.divide(oMeanEpsLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsLarge[5], yerr = np.minimum(oMeanEpsLargeRange[5], np.sqrt(oMeanEpsLarge[5]), np.divide(oMeanEpsLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsLarge[7], yerr = np.minimum(oMeanEpsLargeRange[7], np.sqrt(oMeanEpsLarge[7]), np.divide(oMeanEpsLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_eps_est_large_ord_eps_large.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (small KLD, random +/-)
plt.plot(epsset, rMeanPerc[0], yerr = np.minimum(rMeanPercRange[0], np.sqrt(rMeanPerc[0]), np.divide(rMeanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rMeanPerc[1], yerr = np.minimum(rMeanPercRange[1], np.sqrt(rMeanPerc[1]), np.divide(rMeanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rMeanPerc[2], yerr = np.minimum(rMeanPercRange[2], np.sqrt(rMeanPerc[2]), np.divide(rMeanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([20, 100, 600])
plt.ylim(20, 600)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Synth_eps_perc_small_rand.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (small KLD, ordered)
plt.plot(epsset, oMeanPerc[0], yerr = np.minimum(oMeanPercRange[0], np.sqrt(rMeanPerc[0]), np.divide(rMeanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oMeanPerc[1], yerr = np.minimum(oMeanPercRange[1], np.sqrt(rMeanPerc[1]), np.divide(rMeanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oMeanPerc[2], yerr = np.minimum(oMeanPercRange[2], np.sqrt(rMeanPerc[2]), np.divide(rMeanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([10, 100, 1000])
plt.ylim(8, 1400)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Synth_eps_perc_small_ord.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (large KLD, random +/-)
plt.plot(epsset, rMeanPerc[3], yerr = np.minimum(rMeanPercRange[3], np.sqrt(rMeanPerc[3]), np.divide(rMeanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rMeanPerc[4], yerr = np.minimum(rMeanPercRange[4], np.sqrt(rMeanPerc[4]), np.divide(rMeanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rMeanPerc[5], yerr = np.minimum(rMeanPercRange[5], np.sqrt(rMeanPerc[5]), np.divide(rMeanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([10, 100, 1000, 9000])
plt.ylim(8, 9000)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Synth_eps_perc_large_rand.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (large KLD, ordered)
plt.plot(epsset, oMeanPerc[3], yerr = np.minimum(oMeanPercRange[3], np.sqrt(rMeanPerc[3]), np.divide(rMeanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oMeanPerc[4], yerr = np.minimum(oMeanPercRange[4], np.sqrt(rMeanPerc[4]), np.divide(rMeanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oMeanPerc[5], yerr = np.minimum(oMeanPercRange[5], np.sqrt(rMeanPerc[5]), np.divide(rMeanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([1, 10, 100, 1000])
plt.ylim(1, 1000)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Synth_eps_perc_large_ord.png")
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
