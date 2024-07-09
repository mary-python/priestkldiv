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
Cset = [40, 80, 120, 160, 200, 260, 320, 400, 480, 560, 620, 680]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist_small", "TAgg_small", "Trusted_small", "Dist_large", "TAgg_large", "Trusted_large", "no_privacy_small", "no_privacy_large"]
CS = len(Cset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates (random +/-)
rMeanEst = np.zeros((TS, CS))
rLdaOpt = np.zeros((TS, CS))
rMeanEstZero = np.zeros((TS, CS))
rMeanEstOne = np.zeros((TS, CS))
rMeanPerc = np.zeros((TS, CS))
rMeanCSmall = np.zeros((TS, LS))
rMeanCDef = np.zeros((TS, LS))
rMeanCMid = np.zeros((TS, LS))
rMeanCLarge = np.zeros((TS, LS))

rMeanEstRange = np.zeros((TS, CS))
rLdaOptRange = np.zeros((TS, CS))
rMeanEstZeroRange = np.zeros((TS, CS))
rMeanEstOneRange = np.zeros((TS, CS))
rMeanPercRange = np.zeros((TS, CS))
rMeanCSmallRange = np.zeros((TS, LS))
rMeanCDefRange = np.zeros((TS, LS))
rMeanCMidRange = np.zeros((TS, LS))
rMeanCLargeRange = np.zeros((TS, LS))

# to store statistics related to mean estimates (ordered)
oMeanEst = np.zeros((TS, CS))
oLdaOpt = np.zeros((TS, CS))
oMeanEstZero = np.zeros((TS, CS))
oMeanEstOne = np.zeros((TS, CS))
oMeanPerc = np.zeros((TS, CS))
oMeanCSmall = np.zeros((TS, LS))
oMeanCDef = np.zeros((TS, LS))
oMeanCMid = np.zeros((TS, LS))
oMeanCLarge = np.zeros((TS, LS))

oMeanEstRange = np.zeros((TS, CS))
oLdaOptRange = np.zeros((TS, CS))
oMeanEstZeroRange = np.zeros((TS, CS))
oMeanEstOneRange = np.zeros((TS, CS))
oMeanPercRange = np.zeros((TS, CS))
oMeanCSmallRange = np.zeros((TS, LS))
oMeanCDefRange = np.zeros((TS, LS))
oMeanCMidRange = np.zeros((TS, LS))
oMeanCLargeRange = np.zeros((TS, LS))

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
    randfile = open(f"synth_C_{trialset[trial]}_rand.txt", "w", encoding = 'utf-8')
    ordfile = open(f"synth_C_{trialset[trial]}_ord.txt", "w", encoding = 'utf-8')

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

        rTempMeanEst = np.zeros(RS)
        rTempLdaOpt = np.zeros(RS)
        rTempMeanEstZero = np.zeros(RS)
        rTempMeanEstOne = np.zeros(RS)
        rTempMeanPerc = np.zeros(RS)
        rTempMeanCSmall = np.zeros((LS, RS))
        rTempMeanCDef = np.zeros((LS, RS))
        rTempMeanCMid = np.zeros((LS, RS))
        rTempMeanCLarge = np.zeros((LS, RS))

        oTempMeanEst = np.zeros(RS)
        oTempLdaOpt = np.zeros(RS)
        oTempMeanEstZero = np.zeros(RS)
        oTempMeanEstOne = np.zeros(RS)
        oTempMeanPerc = np.zeros(RS)
        oTempMeanCSmall = np.zeros((LS, RS))
        oTempMeanCDef = np.zeros((LS, RS))
        oTempMeanCMid = np.zeros((LS, RS))
        oTempMeanCLarge = np.zeros((LS, RS))
        
        for rep in range(RS):
            print(f"C = {C}, repeat = {rep + 1}...")
        
            # initialising seed for random sampling
            torch.manual_seed(SEED_FREQ)

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
                    oLogr = oLogr + oStartSample

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
            
                # C = 40 (small)
                if C_COUNT == SMALL_INDEX:
                    rTempMeanCSmall[l, rep] = rMeanLda[l]
                    oTempMeanCSmall[l, rep] = oMeanLda[l]

                # C = 200 (default)
                if C_COUNT == DEF_INDEX:
                    rTempMeanCDef[l, rep] = rMeanLda[l]
                    oTempMeanCDef[l, rep] = oMeanLda[l]

                # C = 400 (mid)
                if C_COUNT == MID_INDEX:
                    rTempMeanCMid[l, rep] = rMeanLda[l]
                    oTempMeanCMid[l, rep] = oMeanLda[l]

                # C = 620 (large)
                if C_COUNT == LARGE_INDEX:
                    rTempMeanCLarge[l, rep] = rMeanLda[l]
                    oTempMeanCLarge[l, rep] = oMeanLda[l]

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
            
                    # C = 40 (small)
                    if C_COUNT == 0:
                        rMeanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanCSmall[l, rep] = (rTempMeanCSmall[l, rep] + rMeanSmallNoise - groundTruth)**2
                        oTempMeanCSmall[l, rep] = (oTempMeanCSmall[l, rep] + oMeanSmallNoise - groundTruth)**2

                    # C = 200 (def)
                    if C_COUNT == 4:
                        rMeanDefNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanDefNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanCDef[l, rep] = (rTempMeanCDef[l, rep] + rMeanDefNoise - groundTruth)**2
                        oTempMeanCDef[l, rep] = (oTempMeanCDef[l, rep] + oMeanDefNoise - groundTruth)**2

                    # C = 400 (mid)
                    if C_COUNT == 7:
                        rMeanMidNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanMidNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanCMid[l, rep] = (rTempMeanCMid[l, rep] + rMeanMidNoise - groundTruth)**2
                        oTempMeanCMid[l, rep] = (oTempMeanCMid[l, rep] + oMeanMidNoise - groundTruth)**2

                    # C = 620 (large)
                    if C_COUNT == 10:
                        rMeanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        oMeanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        rTempMeanCLarge[l, rep] = (rTempMeanCLarge[l, rep] + rMeanLargeNoise - groundTruth)**2
                        oTempMeanCLarge[l, rep] = (oTempMeanCLarge[l, rep] + oMeanLargeNoise - groundTruth)**2

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
            
                    # C = 40 (small)
                    if C_COUNT == 0:
                        rTempMeanCSmall[l, rep] = (rTempMeanCSmall[l, rep] - groundTruth)**2
                        oTempMeanCSmall[l, rep] = (oTempMeanCSmall[l, rep] - groundTruth)**2

                    # C = 200 (def)
                    if C_COUNT == 4:
                        rTempMeanCDef[l, rep] = (rTempMeanCDef[l, rep] - groundTruth)**2
                        oTempMeanCDef[l, rep] = (oTempMeanCDef[l, rep] - groundTruth)**2

                    # C = 400 (mid)
                    if C_COUNT == 7:
                        rTempMeanCMid[l, rep] = (rTempMeanCMid[l, rep] - groundTruth)**2
                        oTempMeanCMid[l, rep] = (oTempMeanCMid[l, rep] - groundTruth)**2

                    # C = 620 (large)
                    if C_COUNT == 10:
                        rTempMeanCLarge[l, rep] = (rTempMeanCLarge[l, rep] - groundTruth)**2
                        oTempMeanCLarge[l, rep] = (oTempMeanCLarge[l, rep] - groundTruth)**2

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
        rMeanEst[trial, C_COUNT] = np.mean(rTempMeanEst)
        rLdaOpt[trial, C_COUNT] = np.mean(rTempLdaOpt)
        rMeanEstZero[trial, C_COUNT] = np.mean(rTempMeanEstZero)
        rMeanEstOne[trial, C_COUNT] = np.mean(rTempMeanEstOne)
        rMeanPerc[trial, C_COUNT] = np.mean(rTempMeanPerc)

        for l in range(LS):
            if C_COUNT == SMALL_INDEX:
                rMeanCSmall[trial, l] = np.mean(rTempMeanCSmall[l])
            if C_COUNT == DEF_INDEX:
                rMeanCDef[trial, l] = np.mean(rTempMeanCDef[l])
            if C_COUNT == MID_INDEX:
                rMeanCMid[trial, l] = np.mean(rTempMeanCMid[l])
            if C_COUNT == LARGE_INDEX:
                rMeanCLarge[trial, l] = np.mean(rTempMeanCLarge[l])

        oMeanEst[trial, C_COUNT] = np.mean(oTempMeanEst)
        oLdaOpt[trial, C_COUNT] = np.mean(oTempLdaOpt)
        oMeanEstZero[trial, C_COUNT] = np.mean(oTempMeanEstZero)
        oMeanEstOne[trial, C_COUNT] = np.mean(oTempMeanEstOne)
        oMeanPerc[trial, C_COUNT] = np.mean(oTempMeanPerc)

        for l in range(LS):
            if C_COUNT == SMALL_INDEX:
                oMeanCSmall[trial, l] = np.mean(oTempMeanCSmall[l])
            if C_COUNT == DEF_INDEX:
                oMeanCDef[trial, l] = np.mean(oTempMeanCDef[l])
            if C_COUNT == MID_INDEX:
                oMeanCMid[trial, l] = np.mean(oTempMeanCMid[l])
            if C_COUNT == LARGE_INDEX:
                oMeanCLarge[trial, l] = np.mean(oTempMeanCLarge[l])
        
        # compute standard deviation of repeats
        rMeanEstRange[trial, C_COUNT] = np.std(rTempMeanEst)
        rLdaOptRange[trial, C_COUNT] = np.std(rTempLdaOpt)
        rMeanEstZeroRange[trial, C_COUNT] = np.std(rTempMeanEstZero)
        rMeanEstOneRange[trial, C_COUNT] = np.std(rTempMeanEstOne)
        rMeanPercRange[trial, C_COUNT] = np.std(rTempMeanPerc)

        for l in range(LS):
            if C_COUNT == SMALL_INDEX:
                rMeanCSmallRange[trial, l] = np.std(rTempMeanCSmall[l])
            if C_COUNT == DEF_INDEX:
                rMeanCDefRange[trial, l] = np.std(rTempMeanCDef[l])
            if C_COUNT == MID_INDEX:
                rMeanCMidRange[trial, l] = np.std(rTempMeanCMid[l])
            if C_COUNT == LARGE_INDEX:
                rMeanCLargeRange[trial, l] = np.std(rTempMeanCLarge[l])

        oMeanEstRange[trial, C_COUNT] = np.std(oTempMeanEst)
        oLdaOptRange[trial, C_COUNT] = np.std(oTempLdaOpt)
        oMeanEstZeroRange[trial, C_COUNT] = np.std(oTempMeanEstZero)
        oMeanEstOneRange[trial, C_COUNT] = np.std(oTempMeanEstOne)
        oMeanPercRange[trial, C_COUNT] = np.std(oTempMeanPerc)

        for l in range(LS):
            if C_COUNT == SMALL_INDEX:
                oMeanCSmallRange[trial, l] = np.std(oTempMeanCSmall[l])
            if C_COUNT == DEF_INDEX:
                oMeanCDefRange[trial, l] = np.std(oTempMeanCDef[l])
            if C_COUNT == MID_INDEX:
                oMeanCMidRange[trial, l] = np.std(oTempMeanCMid[l])
            if C_COUNT == LARGE_INDEX:
                oMeanCLargeRange[trial, l] = np.std(oTempMeanCLarge[l])

        # write statistics on data files
        if C == Cset[0]:
            randfile.write(f"SYNTHETIC Random +/-: C = {C}\n")
            ordfile.write(f"SYNTHETIC Ordered: C = {C}\n")
        else:
            randfile.write(f"\nC = {C}\n")
            ordfile.write(f"\nC = {C}\n")

        randfile.write(f"\nMean Error: {round(rMeanEst[trial, C_COUNT], 2)}\n")
        randfile.write(f"Optimal Lambda: {round(rLdaOpt[trial, C_COUNT], 2)}\n")
        randfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        randfile.write(f"Noise: {np.round(rMeanPerc[trial, C_COUNT], 2)}%\n")

        ordfile.write(f"\nMean Error: {round(oMeanEst[trial, C_COUNT], 2)}\n")
        ordfile.write(f"Optimal Lambda: {round(oLdaOpt[trial, C_COUNT], 2)}\n")
        ordfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")
        ordfile.write(f"Noise: {np.round(oMeanPerc[trial, C_COUNT], 2)}%\n")

        C_COUNT = C_COUNT + 1

# plot error of PRIEST-KLD for each C (small KLD, random +/-)
plt.errorbar(Cset, rMeanEst[0], yerr = np.minimum(rMeanEstRange[0], np.sqrt(rMeanEst[0]), np.divide(rMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEst[1], yerr = np.minimum(rMeanEstRange[1], np.sqrt(rMeanEst[1]), np.divide(rMeanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEst[2], yerr = np.minimum(rMeanEstRange[2], np.sqrt(rMeanEst[2]), np.divide(rMeanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each C (small KLD, ordered)
plt.errorbar(Cset, oMeanEst[0], yerr = np.minimum(oMeanEstRange[0], np.sqrt(oMeanEst[0]), np.divide(oMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEst[1], yerr = np.minimum(oMeanEstRange[1], np.sqrt(oMeanEst[1]), np.divide(oMeanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEst[2], yerr = np.minimum(oMeanEstRange[2], np.sqrt(oMeanEst[2]), np.divide(oMeanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD, random +/-)
plt.errorbar(Cset, rMeanEst[3], yerr = np.minimum(rMeanEstRange[3], np.sqrt(rMeanEst[3]), np.divide(rMeanEst[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEst[4], yerr = np.minimum(rMeanEstRange[4], np.sqrt(rMeanEst[4]), np.divide(rMeanEst[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEst[5], yerr = np.minimum(rMeanEstRange[5], np.sqrt(rMeanEst[5]), np.divide(rMeanEst[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD, ordered)
plt.errorbar(Cset, oMeanEst[3], yerr = np.minimum(oMeanEstRange[3], np.sqrt(oMeanEst[3]), np.divide(oMeanEst[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEst[4], yerr = np.minimum(oMeanEstRange[4], np.sqrt(oMeanEst[4]), np.divide(oMeanEst[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEst[5], yerr = np.minimum(oMeanEstRange[5], np.sqrt(oMeanEst[5]), np.divide(oMeanEst[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord.png")
plt.clf()

# plot optimum lambda for each C (small KLD, random +/-)
plt.errorbar(Cset, rLdaOpt[0], yerr = np.minimum(rLdaOptRange[0], np.sqrt(rLdaOpt[0]), np.divide(rLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rLdaOpt[1], yerr = np.minimum(rLdaOptRange[1], np.sqrt(rLdaOpt[1]), np.divide(rLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rLdaOpt[2], yerr = np.minimum(rLdaOptRange[2], np.sqrt(rLdaOpt[2]), np.divide(rLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_C_lda_opt_small_rand.png")
plt.clf()

# plot optimum lambda for each C (small KLD, ordered)
plt.errorbar(Cset, oLdaOpt[0], yerr = np.minimum(oLdaOptRange[0], np.sqrt(oLdaOpt[0]), np.divide(oLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oLdaOpt[1], yerr = np.minimum(oLdaOptRange[1], np.sqrt(oLdaOpt[1]), np.divide(oLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oLdaOpt[2], yerr = np.minimum(oLdaOptRange[2], np.sqrt(oLdaOpt[2]), np.divide(oLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_C_lda_opt_small_ord.png")
plt.clf()

# plot optimum lambda for each C (large KLD, random +/-)
plt.errorbar(Cset, rLdaOpt[3], yerr = np.minimum(rLdaOptRange[3], np.sqrt(rLdaOpt[3]), np.divide(rLdaOpt[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rLdaOpt[4], yerr = np.minimum(rLdaOptRange[4], np.sqrt(rLdaOpt[4]), np.divide(rLdaOpt[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rLdaOpt[5], yerr = np.minimum(rLdaOptRange[5], np.sqrt(rLdaOpt[5]), np.divide(rLdaOpt[5], 2)),color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_C_lda_opt_large_rand.png")
plt.clf()

# plot optimum lambda for each C (large KLD, ordered)
plt.errorbar(Cset, oLdaOpt[3], yerr = np.minimum(oLdaOptRange[3], np.sqrt(oLdaOpt[3]), np.divide(oLdaOpt[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oLdaOpt[4], yerr = np.minimum(oLdaOptRange[4], np.sqrt(oLdaOpt[4]), np.divide(oLdaOpt[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oLdaOpt[5], yerr = np.minimum(oLdaOptRange[5], np.sqrt(oLdaOpt[5]), np.divide(oLdaOpt[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Synth_C_lda_opt_large_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (small KLD, random +/-)
plt.errorbar(Cset, rMeanEstZero[0], yerr = np.minimum(rMeanEstZeroRange[0], np.sqrt(rMeanEstZero[0]), np.divide(rMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstZero[1], yerr = np.minimum(rMeanEstZeroRange[1], np.sqrt(rMeanEstZero[1]), np.divide(rMeanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstZero[2], yerr = np.minimum(rMeanEstZeroRange[2], np.sqrt(rMeanEstZero[2]), np.divide(rMeanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (small KLD, ordered)
plt.errorbar(Cset, oMeanEstZero[0], yerr = np.minimum(oMeanEstZeroRange[0], np.sqrt(oMeanEstZero[0]), np.divide(oMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstZero[1], yerr = np.minimum(oMeanEstZeroRange[1], np.sqrt(oMeanEstZero[1]), np.divide(oMeanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstZero[2], yerr = np.minimum(oMeanEstZeroRange[2], np.sqrt(oMeanEstZero[2]), np.divide(oMeanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (large KLD, random +/-)
plt.errorbar(Cset, rMeanEstZero[3], yerr = np.minimum(rMeanEstZeroRange[3], np.sqrt(rMeanEstZero[3]), np.divide(rMeanEstZero[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstZero[4], yerr = np.minimum(rMeanEstZeroRange[4], np.sqrt(rMeanEstZero[4]), np.divide(rMeanEstZero[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstZero[5], yerr = np.minimum(rMeanEstZeroRange[5], np.sqrt(rMeanEstZero[5]), np.divide(rMeanEstZero[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (large KLD, ordered)
plt.errorbar(Cset, oMeanEstZero[3], yerr = np.minimum(oMeanEstZeroRange[3], np.sqrt(oMeanEstZero[3]), np.divide(oMeanEstZero[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstZero[4], yerr = np.minimum(oMeanEstZeroRange[4], np.sqrt(oMeanEstZero[4]), np.divide(oMeanEstZero[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstZero[5], yerr = np.minimum(oMeanEstZeroRange[5], np.sqrt(oMeanEstZero[5]), np.divide(oMeanEstZero[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (small KLD, random +/-)
plt.errorbar(Cset, rMeanEstOne[0], yerr = np.minimum(rMeanEstOneRange[0], np.sqrt(rMeanEstOne[0]), np.divide(rMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstOne[1], yerr = np.minimum(rMeanEstOneRange[1], np.sqrt(rMeanEstOne[1]), np.divide(rMeanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstOne[2], yerr = np.minimum(rMeanEstOneRange[2], np.sqrt(rMeanEstOne[2]), np.divide(rMeanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (small KLD, ordered)
plt.errorbar(Cset, oMeanEstOne[0], yerr = np.minimum(oMeanEstOneRange[0], np.sqrt(oMeanEstOne[0]), np.divide(oMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstOne[1], yerr = np.minimum(oMeanEstOneRange[1], np.sqrt(oMeanEstOne[1]), np.divide(oMeanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstOne[2], yerr = np.minimum(oMeanEstOneRange[2], np.sqrt(oMeanEstOne[2]), np.divide(oMeanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (large KLD, random +/-)
plt.errorbar(Cset, rMeanEstOne[3], yerr = np.minimum(rMeanEstOneRange[3], np.sqrt(rMeanEstOne[3]), np.divide(rMeanEstOne[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstOne[4], yerr = np.minimum(rMeanEstOneRange[4], np.sqrt(rMeanEstOne[4]), np.divide(rMeanEstOne[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstOne[5], yerr = np.minimum(rMeanEstOneRange[5], np.sqrt(rMeanEstOne[5]), np.divide(rMeanEstOne[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (large KLD, ordered)
plt.errorbar(Cset, oMeanEstOne[3], yerr = np.minimum(oMeanEstOneRange[3], np.sqrt(oMeanEstOne[3]), np.divide(oMeanEstOne[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstOne[4], yerr = np.minimum(oMeanEstOneRange[4], np.sqrt(oMeanEstOne[4]), np.divide(oMeanEstOne[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstOne[5], yerr = np.minimum(oMeanEstOneRange[5], np.sqrt(oMeanEstOne[5]), np.divide(oMeanEstOne[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when C = 40 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCSmall[0], yerr = np.minimum(rMeanCSmallRange[0], np.sqrt(rMeanCSmall[0]), np.divide(rMeanCSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCSmall[1], yerr = np.minimum(rMeanCSmallRange[1], np.sqrt(rMeanCSmall[1]), np.divide(rMeanCSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCSmall[2], yerr = np.minimum(rMeanCSmallRange[2], np.sqrt(rMeanCSmall[2]), np.divide(rMeanCSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCSmall[6], yerr = np.minimum(rMeanCSmallRange[6], np.sqrt(rMeanCSmall[6]), np.divide(rMeanCSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = 40 (small KLD, ordered)
plt.errorbar(ldaset, oMeanCSmall[0], yerr = np.minimum(oMeanCSmallRange[0], np.sqrt(oMeanCSmall[0]), np.divide(oMeanCSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCSmall[1], yerr = np.minimum(oMeanCSmallRange[1], np.sqrt(oMeanCSmall[1]), np.divide(oMeanCSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCSmall[2], yerr = np.minimum(oMeanCSmallRange[2], np.sqrt(oMeanCSmall[2]), np.divide(oMeanCSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCSmall[6], yerr = np.minimum(oMeanCSmallRange[6], np.sqrt(oMeanCSmall[6]), np.divide(oMeanCSmall[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = 40 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCSmall[3], yerr = np.minimum(rMeanCSmallRange[3], np.sqrt(rMeanCSmall[3]), np.divide(rMeanCSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCSmall[4], yerr = np.minimum(rMeanCSmallRange[4], np.sqrt(rMeanCSmall[4]), np.divide(rMeanCSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCSmall[5], yerr = np.minimum(rMeanCSmallRange[5], np.sqrt(rMeanCSmall[5]), np.divide(rMeanCSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCSmall[7], yerr = np.minimum(rMeanCSmallRange[7], np.sqrt(rMeanCSmall[7]), np.divide(rMeanCSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = 40 (large KLD, ordered)
plt.errorbar(ldaset, oMeanCSmall[3], yerr = np.minimum(oMeanCSmallRange[3], np.sqrt(oMeanCSmall[3]), np.divide(oMeanCSmall[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCSmall[4], yerr = np.minimum(oMeanCSmallRange[4], np.sqrt(oMeanCSmall[4]), np.divide(oMeanCSmall[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCSmall[5], yerr = np.minimum(oMeanCSmallRange[5], np.sqrt(oMeanCSmall[5]), np.divide(oMeanCSmall[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCSmall[7], yerr = np.minimum(oMeanCSmallRange[7], np.sqrt(oMeanCSmall[7]), np.divide(oMeanCSmall[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = 200 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCDef[0], yerr = np.minimum(rMeanCDefRange[0], np.sqrt(rMeanCDef[0]), np.divide(rMeanCDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCDef[1], yerr = np.minimum(rMeanCDefRange[1], np.sqrt(rMeanCDef[1]), np.divide(rMeanCDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCDef[2], yerr = np.minimum(rMeanCDefRange[2], np.sqrt(rMeanCDef[2]), np.divide(rMeanCDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCDef[6], yerr = np.minimum(rMeanCDefRange[6], np.sqrt(rMeanCDef[6]), np.divide(rMeanCDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = 200 (small KLD, ordered)
plt.errorbar(ldaset, oMeanCDef[0], yerr = np.minimum(oMeanCDefRange[0], np.sqrt(oMeanCDef[0]), np.divide(oMeanCDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCDef[1], yerr = np.minimum(oMeanCDefRange[1], np.sqrt(oMeanCDef[1]), np.divide(oMeanCDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCDef[2], yerr = np.minimum(oMeanCDefRange[2], np.sqrt(oMeanCDef[2]), np.divide(oMeanCDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCDef[6], yerr = np.minimum(oMeanCDefRange[6], np.sqrt(oMeanCDef[6]), np.divide(oMeanCDef[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = 200 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCDef[3], yerr = np.minimum(rMeanCDefRange[3], np.sqrt(rMeanCDef[3]), np.divide(rMeanCDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCDef[4], yerr = np.minimum(rMeanCDefRange[4], np.sqrt(rMeanCDef[4]), np.divide(rMeanCDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCDef[5], yerr = np.minimum(rMeanCDefRange[5], np.sqrt(rMeanCDef[5]), np.divide(rMeanCDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCDef[7], yerr = np.minimum(rMeanCDefRange[7], np.sqrt(rMeanCDef[7]), np.divide(rMeanCDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = 200 (large KLD, ordered)
plt.errorbar(ldaset, oMeanCDef[3], yerr = np.minimum(oMeanCDefRange[3], np.sqrt(oMeanCDef[3]), np.divide(oMeanCDef[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCDef[4], yerr = np.minimum(oMeanCDefRange[4], np.sqrt(oMeanCDef[4]), np.divide(oMeanCDef[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCDef[5], yerr = np.minimum(oMeanCDefRange[5], np.sqrt(oMeanCDef[5]), np.divide(oMeanCDef[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCDef[7], yerr = np.minimum(oMeanCDefRange[7], np.sqrt(oMeanCDef[7]), np.divide(oMeanCDef[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = 400 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCMid[0], yerr = np.minimum(rMeanCMidRange[0], np.sqrt(rMeanCMid[0]), np.divide(rMeanCMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCMid[1], yerr = np.minimum(rMeanCMidRange[1], np.sqrt(rMeanCMid[1]), np.divide(rMeanCMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCMid[2], yerr = np.minimum(rMeanCMidRange[2], np.sqrt(rMeanCMid[2]), np.divide(rMeanCMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCMid[6], yerr = np.minimum(rMeanCMidRange[6], np.sqrt(rMeanCMid[6]), np.divide(rMeanCMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = 400 (small KLD, ordered)
plt.errorbar(ldaset, oMeanCMid[0], yerr = np.minimum(oMeanCMidRange[0], np.sqrt(oMeanCMid[0]), np.divide(oMeanCMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCMid[1], yerr = np.minimum(oMeanCMidRange[1], np.sqrt(oMeanCMid[1]), np.divide(oMeanCMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCMid[2], yerr = np.minimum(oMeanCMidRange[2], np.sqrt(oMeanCMid[2]), np.divide(oMeanCMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCMid[6], yerr = np.minimum(oMeanCMidRange[6], np.sqrt(oMeanCMid[6]), np.divide(oMeanCMid[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = 400 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCMid[3], yerr = np.minimum(rMeanCMidRange[3], np.sqrt(rMeanCMid[3]), np.divide(rMeanCMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCMid[4], yerr = np.minimum(rMeanCMidRange[4], np.sqrt(rMeanCMid[4]), np.divide(rMeanCMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCMid[5], yerr = np.minimum(rMeanCMidRange[5], np.sqrt(rMeanCMid[5]), np.divide(rMeanCMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCMid[7], yerr = np.minimum(rMeanCMidRange[7], np.sqrt(rMeanCMid[7]), np.divide(rMeanCMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = 400 (large KLD, ordered)
plt.errorbar(ldaset, oMeanCMid[3], yerr = np.minimum(oMeanCMidRange[3], np.sqrt(oMeanCMid[3]), np.divide(oMeanCMid[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCMid[4], yerr = np.minimum(oMeanCMidRange[4], np.sqrt(oMeanCMid[4]), np.divide(oMeanCMid[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCMid[5], yerr = np.minimum(oMeanCMidRange[5], np.sqrt(oMeanCMid[5]), np.divide(oMeanCMid[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCMid[7], yerr = np.minimum(oMeanCMidRange[7], np.sqrt(oMeanCMid[7]), np.divide(oMeanCMid[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = 620 (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCLarge[0], yerr = np.minimum(rMeanCLargeRange[0], np.sqrt(rMeanCLarge[0]), np.divide(rMeanCLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCLarge[1], yerr = np.minimum(rMeanCLargeRange[1], np.sqrt(rMeanCLarge[1]), np.divide(rMeanCLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCLarge[2], yerr = np.minimum(rMeanCLargeRange[2], np.sqrt(rMeanCLarge[2]), np.divide(rMeanCLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCLarge[6], yerr = np.minimum(rMeanCLargeRange[6], np.sqrt(rMeanCLarge[6]), np.divide(rMeanCLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_rand_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = 620 (small KLD, ordered)
plt.errorbar(ldaset, oMeanCLarge[0], yerr = np.minimum(oMeanCLargeRange[0], np.sqrt(oMeanCLarge[0]), np.divide(oMeanCLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCLarge[1], yerr = np.minimum(oMeanCLargeRange[1], np.sqrt(oMeanCLarge[1]), np.divide(oMeanCLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCLarge[2], yerr = np.minimum(oMeanCLargeRange[2], np.sqrt(oMeanCLarge[2]), np.divide(oMeanCLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCLarge[6], yerr = np.minimum(oMeanCLargeRange[6], np.sqrt(oMeanCLarge[6]), np.divide(oMeanCLarge[6], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_small_ord_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = 620 (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCLarge[3], yerr = np.minimum(rMeanCLargeRange[3], np.sqrt(rMeanCLarge[3]), np.divide(rMeanCLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCLarge[4], yerr = np.minimum(rMeanCLargeRange[4], np.sqrt(rMeanCLarge[4]), np.divide(rMeanCLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCLarge[5], yerr = np.minimum(rMeanCLargeRange[5], np.sqrt(rMeanCLarge[5]), np.divide(rMeanCLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCLarge[7], yerr = np.minimum(rMeanCLargeRange[7], np.sqrt(rMeanCLarge[7]), np.divide(rMeanCLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_rand_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = 620 (large KLD, ordered)
plt.errorbar(ldaset, oMeanCLarge[3], yerr = np.minimum(oMeanCLargeRange[3], np.sqrt(oMeanCLarge[3]), np.divide(oMeanCLarge[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCLarge[4], yerr = np.minimum(oMeanCLargeRange[4], np.sqrt(oMeanCLarge[4]), np.divide(oMeanCLarge[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCLarge[5], yerr = np.minimum(oMeanCLargeRange[5], np.sqrt(oMeanCLarge[5]), np.divide(oMeanCLarge[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCLarge[7], yerr = np.minimum(oMeanCLargeRange[7], np.sqrt(oMeanCLarge[7]), np.divide(oMeanCLarge[7], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Synth_C_est_large_ord_C_large.png")
plt.clf()

# plot % of noise vs ground truth for each C (small KLD, random +/-)
plt.errorbar(Cset, rMeanPerc[0], yerr = np.minimum(rMeanPercRange[0], np.sqrt(rMeanPerc[0]), np.divide(rMeanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanPerc[1], yerr = np.minimum(rMeanPercRange[1], np.sqrt(rMeanPerc[1]), np.divide(rMeanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanPerc[2], yerr = np.minimum(rMeanPercRange[2], np.sqrt(rMeanPerc[2]), np.divide(rMeanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([60, 100, 130])
plt.ylim(60, 130)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.savefig("Synth_C_perc_small_rand.png")
plt.clf()

# plot % of noise vs ground truth for each C (small KLD, ordered)
plt.errorbar(Cset, oMeanPerc[0], yerr = np.minimum(oMeanPercRange[0], np.sqrt(rMeanPerc[0]), np.divide(rMeanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanPerc[1], yerr = np.minimum(oMeanPercRange[1], np.sqrt(rMeanPerc[1]), np.divide(rMeanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanPerc[2], yerr = np.minimum(oMeanPercRange[2], np.sqrt(rMeanPerc[2]), np.divide(rMeanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([50, 100, 500])
plt.ylim(50, 500)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.savefig("Synth_C_perc_small_ord.png")
plt.clf()

# plot % of noise vs ground truth for each C (large KLD, random +/-)
plt.errorbar(Cset, rMeanPerc[3], yerr = np.minimum(rMeanPercRange[3], np.sqrt(rMeanPerc[3]), np.divide(rMeanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanPerc[4], yerr = np.minimum(rMeanPercRange[4], np.sqrt(rMeanPerc[4]), np.divide(rMeanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanPerc[5], yerr = np.minimum(rMeanPercRange[5], np.sqrt(rMeanPerc[5]), np.divide(rMeanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([40, 100, 500])
plt.ylim(35, 550)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.savefig("Synth_C_perc_large_rand.png")
plt.clf()

# plot % of noise vs ground truth for each C (large KLD, ordered)
plt.errorbar(Cset, oMeanPerc[3], yerr = np.minimum(oMeanPercRange[3], np.sqrt(rMeanPerc[3]), np.divide(rMeanPerc[3], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanPerc[4], yerr = np.minimum(oMeanPercRange[4], np.sqrt(rMeanPerc[4]), np.divide(rMeanPerc[4], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanPerc[5], yerr = np.minimum(oMeanPercRange[5], np.sqrt(rMeanPerc[5]), np.divide(rMeanPerc[5], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([25, 100, 800])
plt.ylim(25, 800)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.savefig("Synth_C_perc_large_ord.png")
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