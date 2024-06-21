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

# lists of the values of C and lambda, as well as the trials that will be explored
Cset = [40, 80, 120, 160, 200, 260, 320, 400, 480, 560, 620, 680]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist_small", "Dist_small_mc", "TAgg_small", "TAgg_small_mc", "Trusted_small", "Trusted_small_mc", "Dist_large",
            "Dist_large_mc", "TAgg_large", "TAgg_large_mc", "Trusted_large", "Trusted_large_mc", "NoAlgo_small", "NoAlgo_large"]
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

for trial in range(14):
    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    randfile = open(f"synth_C_{trialset[trial]}_rand.txt", "w", encoding = 'utf-8')
    ordfile = open(f"synth_C_{trialset[trial]}_ord.txt", "w", encoding = 'utf-8')

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial < 6 or trial == 12:
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

    # each client has 125 points
    N = 125

    # parameters for the addition of Laplace and Gaussian noise
    EPS = 0.5
    DTA = 0.1
    A = 0
    R = 10

    if trial < 12:

        # option 2a: baseline case
        if trial % 2 == 0:
            b1 = log(2)
    
        # option 2b: Monte-Carlo estimate
        else:
            b1 = 1 + log(2)

        b2 = 2*((log(1.25))/DTA)*b1

    if trial < 12:
    
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
        print(f"Trial {trial + 1}: C = {C}...")

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

            # option 3a: "Dist" (each client adds Gaussian noise term)
            if trial % 3 == 0 and trial != 12:
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
                rRangeEst = (lda * (np.exp(rLogr) - 1)) - rLogr
                oRangeEst = (lda * (np.exp(oLogr) - 1)) - oLogr

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

            # option 3b: "TAgg" (intermediate server adds Gaussian noise term)
            if trial % 3 == 1 and trial != 13:
                rMeanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                oMeanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                rMeanLda[l] = rMeanLda[l] + rMeanLdaNoise[l]
                oMeanLda[l] = oMeanLda[l] + oMeanLdaNoise[l]
            
            # C = 40 (small)
            if C_COUNT == 0:
                rMeanCSmall[trial, l] = rMeanLda[l]
                oMeanCSmall[trial, l] = oMeanLda[l]

            # C = 200 (default)
            if C_COUNT == 4:
                rMeanCDef[trial, l] = rMeanLda[l]
                oMeanCDef[trial, l] = oMeanLda[l]

            # C = 400 (mid)
            if C_COUNT == 7:
                rMeanCMid[trial, l] = rMeanLda[l]
                oMeanCMid[trial, l] = oMeanLda[l]

            # C = 620 (large)
            if C_COUNT == 10:
                rMeanCLarge[trial, l] = rMeanLda[l]
                oMeanCLarge[trial, l] = oMeanLda[l]

        # find lambda that produces minimum error
        rLdaIndex = np.argmin(rMeanLda)
        oLdaIndex = np.argmin(oMeanLda)

        rMinMeanError = rMeanLda[rLdaIndex]
        oMinMeanError = oMeanLda[oLdaIndex]

        # mean across clients for optimum lambda
        rMeanEst[trial, C_COUNT] = rMinMeanError
        oMeanEst[trial, C_COUNT] = oMinMeanError

        # optimum lambda
        ldaStep = 0.05
        rLdaOpt[trial, C_COUNT] = rLdaIndex * ldaStep
        oLdaOpt[trial, C_COUNT] = oLdaIndex * ldaStep

        # lambda = 0
        rMeanEstZero[trial, C_COUNT] = rMeanLda[0]
        oMeanEstZero[trial, C_COUNT] = oMeanLda[0]

        # lambda = 1
        rMeanEstOne[trial, C_COUNT] = rMeanLda[LS-1]
        oMeanEstOne[trial, C_COUNT] = oMeanLda[LS-1]

        # option 3c: "Trusted" (server adds Laplace noise term to final result)
        if trial % 3 == 2:
            rMeanNoise = lapNoise.sample(sample_shape = (1,))
            oMeanNoise = lapNoise.sample(sample_shape = (1,))
            rMeanZeroNoise = lapNoise.sample(sample_shape = (1,))
            oMeanZeroNoise = lapNoise.sample(sample_shape = (1,))
            rMeanOneNoise = lapNoise.sample(sample_shape = (1,))
            oMeanOneNoise = lapNoise.sample(sample_shape = (1,))

            # define error = squared difference between estimator and ground truth
            rMeanEst[trial, C_COUNT] = (rMeanEst[trial, C_COUNT] + rMeanNoise - groundTruth)**2
            oMeanEst[trial, C_COUNT] = (oMeanEst[trial, C_COUNT] + oMeanNoise - groundTruth)**2

            # lambda = 0
            rMeanEstZero[trial, C_COUNT] = (rMeanEstZero[trial, C_COUNT] + rMeanZeroNoise - groundTruth)**2
            oMeanEstZero[trial, C_COUNT] = (oMeanEstZero[trial, C_COUNT] + oMeanZeroNoise - groundTruth)**2

            # lambda = 1
            rMeanEstOne[trial, C_COUNT] = (rMeanEstOne[trial, C_COUNT] + rMeanOneNoise - groundTruth)**2
            oMeanEstOne[trial, C_COUNT] = (oMeanEstOne[trial, C_COUNT] + oMeanOneNoise - groundTruth)**2

        # clients or intermediate server already added Gaussian noise term
        else:
            rMeanEst[trial, C_COUNT] = (rMeanEst[trial, C_COUNT] - groundTruth)**2
            oMeanEst[trial, C_COUNT] = (oMeanEst[trial, C_COUNT] - groundTruth)**2

            # lambda = 0
            rMeanEstZero[trial, C_COUNT] = (rMeanEstZero[trial, C_COUNT] - groundTruth)**2
            oMeanEstZero[trial, C_COUNT] = (oMeanEstZero[trial, C_COUNT] - groundTruth)**2

            # lambda = 1
            rMeanEstOne[trial, C_COUNT] = (rMeanEstOne[trial, C_COUNT] - groundTruth)**2
            oMeanEstOne[trial, C_COUNT] = (oMeanEstOne[trial, C_COUNT] - groundTruth)**2

        if C == Cset[0]:
            randfile.write(f"SYNTHETIC Random +/-: C = {C}\n")
            ordfile.write(f"SYNTHETIC Ordered: C = {C}\n")
        else:
            randfile.write(f"\nC = {C}\n")
            ordfile.write(f"\nC = {C}\n")

        randfile.write(f"\nMean Error: {round(rMeanEst[trial, C_COUNT], 2)}\n")
        randfile.write(f"Optimal Lambda: {round(rLdaOpt[trial, C_COUNT], 2)}\n")
        randfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")

        # compute % of noise vs ground truth (random +/-)
        if trial % 3 == 0 and trial != 12:
            rMeanPerc[trial, C_COUNT] = float(abs(np.array(sum(rStartNoise)) / (np.array(sum(rStartNoise) + groundTruth))))*100
            randfile.write(f"Noise: {np.round(rMeanPerc[trial, C_COUNT], 2)}%\n")
        if trial % 3 == 1 and trial != 13:
            rMeanPerc[trial, C_COUNT] = abs((np.sum(rMeanLdaNoise)) / (np.sum(rMeanLdaNoise) + groundTruth))*100
            randfile.write(f"Noise: {round(rMeanPerc[trial, C_COUNT], 2)}%\n")
        if trial % 3 == 2:
            rMeanPerc[trial, C_COUNT] = float(abs(np.array(rMeanNoise) / (np.array(rMeanNoise + groundTruth))))*100
            randfile.write(f"Noise: {np.round(rMeanPerc[trial, C_COUNT], 2)}%\n")

        ordfile.write(f"\nMean Error: {round(oMeanEst[trial, C_COUNT], 2)}\n")
        ordfile.write(f"Optimal Lambda: {round(oLdaOpt[trial, C_COUNT], 2)}\n")
        ordfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")

        # compute % of noise vs ground truth (ordered)
        if trial % 3 == 0 and trial != 12:
            oMeanPerc[trial, C_COUNT] = float(abs(np.array(sum(oStartNoise)) / (np.array(sum(oStartNoise) + groundTruth))))*100
            ordfile.write(f"Noise: {np.round(oMeanPerc[trial, C_COUNT], 2)}%\n")
        if trial % 3 == 1 and trial != 13:
            oMeanPerc[trial, C_COUNT] = abs((np.sum(oMeanLdaNoise)) / (np.sum(oMeanLdaNoise) + groundTruth))*100
            ordfile.write(f"Noise: {round(oMeanPerc[trial, C_COUNT], 2)}%\n")
        if trial % 3 == 2:
            oMeanPerc[trial, C_COUNT] = float(abs(np.array(oMeanNoise) / (np.array(oMeanNoise + groundTruth))))*100
            ordfile.write(f"Noise: {np.round(oMeanPerc[trial, C_COUNT], 2)}%\n")

        C_COUNT = C_COUNT + 1

# plot error of PRIEST-KLD for each C (small KLD, random +/-)
plt.errorbar(Cset, rMeanEst[0], yerr = np.minimum(np.sqrt(rMeanEst[0]), np.divide(rMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEst[2], yerr = np.minimum(np.sqrt(rMeanEst[2]), np.divide(rMeanEst[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEst[4], yerr = np.minimum(np.sqrt(rMeanEst[4]), np.divide(rMeanEst[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, rMeanEst[12], yerr = np.minimum(np.sqrt(rMeanEst[12]), np.divide(rMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each C (small KLD, random +/-, mc)
plt.errorbar(Cset, rMeanEst[1], yerr = np.minimum(np.sqrt(rMeanEst[1]), np.divide(rMeanEst[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, rMeanEst[3], yerr = np.minimum(np.sqrt(rMeanEst[3]), np.divide(rMeanEst[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, rMeanEst[5], yerr = np.minimum(np.sqrt(rMeanEst[5]), np.divide(rMeanEst[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, rMeanEst[12], yerr = np.minimum(np.sqrt(rMeanEst[12]), np.divide(rMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc.png")
plt.clf()

# plot error of PRIEST-KLD for each C (small KLD, ordered)
plt.errorbar(Cset, oMeanEst[0], yerr = np.minimum(np.sqrt(oMeanEst[0]), np.divide(oMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEst[2], yerr = np.minimum(np.sqrt(oMeanEst[2]), np.divide(oMeanEst[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEst[4], yerr = np.minimum(np.sqrt(oMeanEst[4]), np.divide(oMeanEst[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, oMeanEst[12], yerr = np.minimum(np.sqrt(oMeanEst[12]), np.divide(oMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord.png")
plt.clf()

# plot error of PRIEST-KLD for each C (small KLD, ordered, mc)
plt.errorbar(Cset, oMeanEst[1], yerr = np.minimum(np.sqrt(oMeanEst[1]), np.divide(oMeanEst[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, oMeanEst[3], yerr = np.minimum(np.sqrt(oMeanEst[3]), np.divide(oMeanEst[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, oMeanEst[5], yerr = np.minimum(np.sqrt(oMeanEst[5]), np.divide(oMeanEst[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, oMeanEst[12], yerr = np.minimum(np.sqrt(oMeanEst[12]), np.divide(oMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD, random +/-)
plt.errorbar(Cset, rMeanEst[6], yerr = np.minimum(np.sqrt(rMeanEst[6]), np.divide(rMeanEst[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEst[8], yerr = np.minimum(np.sqrt(rMeanEst[8]), np.divide(rMeanEst[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEst[10], yerr = np.minimum(np.sqrt(rMeanEst[10]), np.divide(rMeanEst[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, rMeanEst[13], yerr = np.minimum(np.sqrt(rMeanEst[13]), np.divide(rMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD, random +/-, mc)
plt.errorbar(Cset, rMeanEst[7], yerr = np.minimum(np.sqrt(rMeanEst[7]), np.divide(rMeanEst[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, rMeanEst[9], yerr = np.minimum(np.sqrt(rMeanEst[9]), np.divide(rMeanEst[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, rMeanEst[11], yerr = np.minimum(np.sqrt(rMeanEst[11]), np.divide(rMeanEst[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, rMeanEst[13], yerr = np.minimum(np.sqrt(rMeanEst[13]), np.divide(rMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD, ordered)
plt.errorbar(Cset, oMeanEst[6], yerr = np.minimum(np.sqrt(oMeanEst[6]), np.divide(oMeanEst[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEst[8], yerr = np.minimum(np.sqrt(oMeanEst[8]), np.divide(oMeanEst[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEst[10], yerr = np.minimum(np.sqrt(oMeanEst[10]), np.divide(oMeanEst[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, oMeanEst[13], yerr = np.minimum(np.sqrt(oMeanEst[13]), np.divide(oMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord.png")
plt.clf()

# plot error of PRIEST-KLD for each C (large KLD, ordered, mc)
plt.errorbar(Cset, oMeanEst[7], yerr = np.minimum(np.sqrt(oMeanEst[7]), np.divide(oMeanEst[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, oMeanEst[9], yerr = np.minimum(np.sqrt(oMeanEst[9]), np.divide(oMeanEst[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, oMeanEst[11], yerr = np.minimum(np.sqrt(oMeanEst[11]), np.divide(oMeanEst[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, oMeanEst[13], yerr = np.minimum(np.sqrt(oMeanEst[13]), np.divide(oMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Error of PRIEST-KLD")
plt.title("C vs error of PRIEST-KLD (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc.png")
plt.clf()

# plot optimum lambda for each C (small KLD, random +/-)
plt.plot(Cset, rLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, rLdaOpt[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, rLdaOpt[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, rLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (small KLD, random +/-)")
plt.savefig("Synth_C_lda_opt_small_rand.png")
plt.clf()

# plot optimum lambda for each C (small KLD, random +/-, mc)
plt.plot(Cset, rLdaOpt[1], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(Cset, rLdaOpt[3], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(Cset, rLdaOpt[5], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(Cset, rLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (small KLD, random +/-, mc)")
plt.savefig("Synth_C_lda_opt_small_rand_mc.png")
plt.clf()

# plot optimum lambda for each C (small KLD, ordered)
plt.plot(Cset, oLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, oLdaOpt[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, oLdaOpt[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, oLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (small KLD, ordered)")
plt.savefig("Synth_C_lda_opt_small_ord.png")
plt.clf()

# plot optimum lambda for each C (small KLD, ordered, mc)
plt.plot(Cset, oLdaOpt[1], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(Cset, oLdaOpt[3], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(Cset, oLdaOpt[5], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(Cset, oLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (small KLD, ordered, mc)")
plt.savefig("Synth_C_lda_opt_small_ord_mc.png")
plt.clf()

# plot optimum lambda for each C (large KLD, random +/-)
plt.plot(Cset, rLdaOpt[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, rLdaOpt[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, rLdaOpt[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, rLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (large KLD, random +/-)")
plt.savefig("Synth_C_lda_opt_large_rand.png")
plt.clf()

# plot optimum lambda for each C (large KLD, random +/-, mc)
plt.plot(Cset, rLdaOpt[7], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(Cset, rLdaOpt[9], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(Cset, rLdaOpt[11], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(Cset, rLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (large KLD, random +/-, mc)")
plt.savefig("Synth_C_lda_opt_large_rand_mc.png")
plt.clf()

# plot optimum lambda for each C (large KLD, ordered)
plt.plot(Cset, oLdaOpt[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, oLdaOpt[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, oLdaOpt[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, oLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (large KLD, ordered)")
plt.savefig("Synth_C_lda_opt_large_ord.png")
plt.clf()

# plot optimum lambda for each C (large KLD, ordered, mc)
plt.plot(Cset, oLdaOpt[7], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(Cset, oLdaOpt[9], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(Cset, oLdaOpt[11], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(Cset, oLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of C")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("C vs optimum lambda (large KLD, ordered, mc)")
plt.savefig("Synth_C_lda_opt_large_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (small KLD, random +/-)
plt.errorbar(Cset, rMeanEstZero[0], yerr = np.minimum(np.sqrt(rMeanEstZero[0]), np.divide(rMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstZero[2], yerr = np.minimum(np.sqrt(rMeanEstZero[2]), np.divide(rMeanEstZero[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstZero[4], yerr = np.minimum(np.sqrt(rMeanEstZero[4]), np.divide(rMeanEstZero[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, rMeanEstZero[12], yerr = np.minimum(np.sqrt(rMeanEstZero[12]), np.divide(rMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (small KLD, random +/-, mc)
plt.errorbar(Cset, rMeanEstZero[1], yerr = np.minimum(np.sqrt(rMeanEstZero[1]), np.divide(rMeanEstZero[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, rMeanEstZero[3], yerr = np.minimum(np.sqrt(rMeanEstZero[3]), np.divide(rMeanEstZero[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, rMeanEstZero[5], yerr = np.minimum(np.sqrt(rMeanEstZero[5]), np.divide(rMeanEstZero[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, rMeanEstZero[12], yerr = np.minimum(np.sqrt(rMeanEstZero[12]), np.divide(rMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (small KLD, ordered)
plt.errorbar(Cset, oMeanEstZero[0], yerr = np.minimum(np.sqrt(oMeanEstZero[0]), np.divide(oMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstZero[2], yerr = np.minimum(np.sqrt(oMeanEstZero[2]), np.divide(oMeanEstZero[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstZero[4], yerr = np.minimum(np.sqrt(oMeanEstZero[4]), np.divide(oMeanEstZero[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, oMeanEstZero[12], yerr = np.minimum(np.sqrt(oMeanEstZero[12]), np.divide(oMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (small KLD, ordered, mc)
plt.errorbar(Cset, oMeanEstZero[1], yerr = np.minimum(np.sqrt(oMeanEstZero[1]), np.divide(oMeanEstZero[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, oMeanEstZero[3], yerr = np.minimum(np.sqrt(oMeanEstZero[3]), np.divide(oMeanEstZero[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, oMeanEstZero[5], yerr = np.minimum(np.sqrt(oMeanEstZero[5]), np.divide(oMeanEstZero[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, oMeanEstZero[12], yerr = np.minimum(np.sqrt(oMeanEstZero[12]), np.divide(oMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (large KLD, random +/-)
plt.errorbar(Cset, rMeanEstZero[6], yerr = np.minimum(np.sqrt(rMeanEstZero[6]), np.divide(rMeanEstZero[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstZero[8], yerr = np.minimum(np.sqrt(rMeanEstZero[8]), np.divide(rMeanEstZero[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstZero[10], yerr = np.minimum(np.sqrt(rMeanEstZero[10]), np.divide(rMeanEstZero[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, rMeanEstZero[13], yerr = np.minimum(np.sqrt(rMeanEstZero[13]), np.divide(rMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (large KLD, random +/-, mc)
plt.errorbar(Cset, rMeanEstZero[7], yerr = np.minimum(np.sqrt(rMeanEstZero[7]), np.divide(rMeanEstZero[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, rMeanEstZero[9], yerr = np.minimum(np.sqrt(rMeanEstZero[9]), np.divide(rMeanEstZero[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, rMeanEstZero[11], yerr = np.minimum(np.sqrt(rMeanEstZero[11]), np.divide(rMeanEstZero[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, rMeanEstZero[13], yerr = np.minimum(np.sqrt(rMeanEstZero[13]), np.divide(rMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (large KLD, ordered)
plt.errorbar(Cset, oMeanEstZero[6], yerr = np.minimum(np.sqrt(oMeanEstZero[6]), np.divide(oMeanEstZero[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstZero[8], yerr = np.minimum(np.sqrt(oMeanEstZero[8]), np.divide(oMeanEstZero[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstZero[10], yerr = np.minimum(np.sqrt(oMeanEstZero[10]), np.divide(oMeanEstZero[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, oMeanEstZero[13], yerr = np.minimum(np.sqrt(oMeanEstZero[13]), np.divide(oMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (large KLD, ordered, mc)
plt.errorbar(Cset, oMeanEstZero[7], yerr = np.minimum(np.sqrt(oMeanEstZero[7]), np.divide(oMeanEstZero[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, oMeanEstZero[9], yerr = np.minimum(np.sqrt(oMeanEstZero[9]), np.divide(oMeanEstZero[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, oMeanEstZero[11], yerr = np.minimum(np.sqrt(oMeanEstZero[11]), np.divide(oMeanEstZero[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, oMeanEstZero[13], yerr = np.minimum(np.sqrt(oMeanEstZero[13]), np.divide(oMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (small KLD, random +/-)
plt.errorbar(Cset, rMeanEstOne[0], yerr = np.minimum(np.sqrt(rMeanEstOne[0]), np.divide(rMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstOne[2], yerr = np.minimum(np.sqrt(rMeanEstOne[2]), np.divide(rMeanEstOne[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstOne[4], yerr = np.minimum(np.sqrt(rMeanEstOne[4]), np.divide(rMeanEstOne[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, rMeanEstOne[12], yerr = np.minimum(np.sqrt(rMeanEstOne[12]), np.divide(rMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (small KLD, random +/-, mc)
plt.errorbar(Cset, rMeanEstOne[1], yerr = np.minimum(np.sqrt(rMeanEstOne[1]), np.divide(rMeanEstOne[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, rMeanEstOne[3], yerr = np.minimum(np.sqrt(rMeanEstOne[3]), np.divide(rMeanEstOne[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, rMeanEstOne[5], yerr = np.minimum(np.sqrt(rMeanEstOne[5]), np.divide(rMeanEstOne[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, rMeanEstOne[12], yerr = np.minimum(np.sqrt(rMeanEstOne[12]), np.divide(rMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (small KLD, ordered)
plt.errorbar(Cset, oMeanEstOne[0], yerr = np.minimum(np.sqrt(oMeanEstOne[0]), np.divide(oMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstOne[2], yerr = np.minimum(np.sqrt(oMeanEstOne[2]), np.divide(oMeanEstOne[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstOne[4], yerr = np.minimum(np.sqrt(oMeanEstOne[4]), np.divide(oMeanEstOne[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, oMeanEstOne[12], yerr = np.minimum(np.sqrt(oMeanEstOne[12]), np.divide(oMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (small KLD, ordered, mc)
plt.errorbar(Cset, oMeanEstOne[1], yerr = np.minimum(np.sqrt(oMeanEstOne[1]), np.divide(oMeanEstOne[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, oMeanEstOne[3], yerr = np.minimum(np.sqrt(oMeanEstOne[3]), np.divide(oMeanEstOne[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, oMeanEstOne[5], yerr = np.minimum(np.sqrt(oMeanEstOne[5]), np.divide(oMeanEstOne[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, oMeanEstOne[12], yerr = np.minimum(np.sqrt(oMeanEstOne[12]), np.divide(oMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (large KLD, random +/-)
plt.errorbar(Cset, rMeanEstOne[6], yerr = np.minimum(np.sqrt(rMeanEstOne[6]), np.divide(rMeanEstOne[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, rMeanEstOne[8], yerr = np.minimum(np.sqrt(rMeanEstOne[8]), np.divide(rMeanEstOne[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, rMeanEstOne[10], yerr = np.minimum(np.sqrt(rMeanEstOne[10]), np.divide(rMeanEstOne[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, rMeanEstOne[13], yerr = np.minimum(np.sqrt(rMeanEstOne[13]), np.divide(rMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (large KLD, random +/-, mc)
plt.errorbar(Cset, rMeanEstOne[7], yerr = np.minimum(np.sqrt(rMeanEstOne[7]), np.divide(rMeanEstOne[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, rMeanEstOne[9], yerr = np.minimum(np.sqrt(rMeanEstOne[9]), np.divide(rMeanEstOne[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, rMeanEstOne[11], yerr = np.minimum(np.sqrt(rMeanEstOne[11]), np.divide(rMeanEstOne[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, rMeanEstOne[13], yerr = np.minimum(np.sqrt(rMeanEstOne[13]), np.divide(rMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (large KLD, ordered)
plt.errorbar(Cset, oMeanEstOne[6], yerr = np.minimum(np.sqrt(oMeanEstOne[6]), np.divide(oMeanEstOne[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Cset, oMeanEstOne[8], yerr = np.minimum(np.sqrt(oMeanEstOne[8]), np.divide(oMeanEstOne[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Cset, oMeanEstOne[10], yerr = np.minimum(np.sqrt(oMeanEstOne[10]), np.divide(oMeanEstOne[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Cset, oMeanEstOne[13], yerr = np.minimum(np.sqrt(oMeanEstOne[13]), np.divide(oMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (large KLD, ordered, mc)
plt.errorbar(Cset, oMeanEstOne[7], yerr = np.minimum(np.sqrt(oMeanEstOne[7]), np.divide(oMeanEstOne[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(Cset, oMeanEstOne[9], yerr = np.minimum(np.sqrt(oMeanEstOne[9]), np.divide(oMeanEstOne[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(Cset, oMeanEstOne[11], yerr = np.minimum(np.sqrt(oMeanEstOne[11]), np.divide(oMeanEstOne[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(Cset, oMeanEstOne[13], yerr = np.minimum(np.sqrt(oMeanEstOne[13]), np.divide(oMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCSmall[0], yerr = np.minimum(np.sqrt(rMeanCSmall[0]), np.divide(rMeanCSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCSmall[2], yerr = np.minimum(np.sqrt(rMeanCSmall[2]), np.divide(rMeanCSmall[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCSmall[4], yerr = np.minimum(np.sqrt(rMeanCSmall[4]), np.divide(rMeanCSmall[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCSmall[12], yerr = np.minimum(np.sqrt(rMeanCSmall[12]), np.divide(rMeanCSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCSmall[1], yerr = np.minimum(np.sqrt(rMeanCSmall[1]), np.divide(rMeanCSmall[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCSmall[3], yerr = np.minimum(np.sqrt(rMeanCSmall[3]), np.divide(rMeanCSmall[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCSmall[5], yerr = np.minimum(np.sqrt(rMeanCSmall[5]), np.divide(rMeanCSmall[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCSmall[12], yerr = np.minimum(np.sqrt(rMeanCSmall[12]), np.divide(rMeanCSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (small KLD, ordered)
plt.errorbar(ldaset, oMeanCSmall[0], yerr = np.minimum(np.sqrt(oMeanCSmall[0]), np.divide(oMeanCSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCSmall[2], yerr = np.minimum(np.sqrt(oMeanCSmall[2]), np.divide(oMeanCSmall[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCSmall[4], yerr = np.minimum(np.sqrt(oMeanCSmall[4]), np.divide(oMeanCSmall[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCSmall[12], yerr = np.minimum(np.sqrt(oMeanCSmall[12]), np.divide(oMeanCSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCSmall[1], yerr = np.minimum(np.sqrt(oMeanCSmall[1]), np.divide(oMeanCSmall[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCSmall[3], yerr = np.minimum(np.sqrt(oMeanCSmall[3]), np.divide(oMeanCSmall[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCSmall[5], yerr = np.minimum(np.sqrt(oMeanCSmall[5]), np.divide(oMeanCSmall[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCSmall[12], yerr = np.minimum(np.sqrt(oMeanCSmall[12]), np.divide(oMeanCSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCSmall[6], yerr = np.minimum(np.sqrt(rMeanCSmall[6]), np.divide(rMeanCSmall[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCSmall[8], yerr = np.minimum(np.sqrt(rMeanCSmall[8]), np.divide(rMeanCSmall[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCSmall[10], yerr = np.minimum(np.sqrt(rMeanCSmall[10]), np.divide(rMeanCSmall[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCSmall[13], yerr = np.minimum(np.sqrt(rMeanCSmall[13]), np.divide(rMeanCSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCSmall[7], yerr = np.minimum(np.sqrt(rMeanCSmall[7]), np.divide(rMeanCSmall[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCSmall[9], yerr = np.minimum(np.sqrt(rMeanCSmall[9]), np.divide(rMeanCSmall[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCSmall[11], yerr = np.minimum(np.sqrt(rMeanCSmall[11]), np.divide(rMeanCSmall[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCSmall[13], yerr = np.minimum(np.sqrt(rMeanCSmall[13]), np.divide(rMeanCSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (large KLD, ordered)
plt.errorbar(ldaset, oMeanCSmall[6], yerr = np.minimum(np.sqrt(oMeanCSmall[6]), np.divide(oMeanCSmall[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCSmall[8], yerr = np.minimum(np.sqrt(oMeanCSmall[8]), np.divide(oMeanCSmall[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCSmall[10], yerr = np.minimum(np.sqrt(oMeanCSmall[10]), np.divide(oMeanCSmall[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCSmall[13], yerr = np.minimum(np.sqrt(oMeanCSmall[13]), np.divide(oMeanCSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = small (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCSmall[7], yerr = np.minimum(np.sqrt(oMeanCSmall[7]), np.divide(oMeanCSmall[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCSmall[9], yerr = np.minimum(np.sqrt(oMeanCSmall[9]), np.divide(oMeanCSmall[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCSmall[11], yerr = np.minimum(np.sqrt(oMeanCSmall[11]), np.divide(oMeanCSmall[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCSmall[13], yerr = np.minimum(np.sqrt(oMeanCSmall[13]), np.divide(oMeanCSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[0]} (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc_C_small.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCDef[0], yerr = np.minimum(np.sqrt(rMeanCDef[0]), np.divide(rMeanCDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCDef[2], yerr = np.minimum(np.sqrt(rMeanCDef[2]), np.divide(rMeanCDef[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCDef[4], yerr = np.minimum(np.sqrt(rMeanCDef[4]), np.divide(rMeanCDef[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCDef[12], yerr = np.minimum(np.sqrt(rMeanCDef[12]), np.divide(rMeanCDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCDef[1], yerr = np.minimum(np.sqrt(rMeanCDef[1]), np.divide(rMeanCDef[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCDef[3], yerr = np.minimum(np.sqrt(rMeanCDef[3]), np.divide(rMeanCDef[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCDef[5], yerr = np.minimum(np.sqrt(rMeanCDef[5]), np.divide(rMeanCDef[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCDef[12], yerr = np.minimum(np.sqrt(rMeanCDef[12]), np.divide(rMeanCDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (small KLD, ordered)
plt.errorbar(ldaset, oMeanCDef[0], yerr = np.minimum(np.sqrt(oMeanCDef[0]), np.divide(oMeanCDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCDef[2], yerr = np.minimum(np.sqrt(oMeanCDef[2]), np.divide(oMeanCDef[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCDef[4], yerr = np.minimum(np.sqrt(oMeanCDef[4]), np.divide(oMeanCDef[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCDef[12], yerr = np.minimum(np.sqrt(oMeanCDef[12]), np.divide(oMeanCDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCDef[1], yerr = np.minimum(np.sqrt(oMeanCDef[1]), np.divide(oMeanCDef[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCDef[3], yerr = np.minimum(np.sqrt(oMeanCDef[3]), np.divide(oMeanCDef[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCDef[5], yerr = np.minimum(np.sqrt(oMeanCDef[5]), np.divide(oMeanCDef[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCDef[12], yerr = np.minimum(np.sqrt(oMeanCDef[12]), np.divide(oMeanCDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCDef[6], yerr = np.minimum(np.sqrt(rMeanCDef[6]), np.divide(rMeanCDef[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCDef[8], yerr = np.minimum(np.sqrt(rMeanCDef[8]), np.divide(rMeanCDef[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCDef[10], yerr = np.minimum(np.sqrt(rMeanCDef[10]), np.divide(rMeanCDef[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCDef[13], yerr = np.minimum(np.sqrt(rMeanCDef[13]), np.divide(rMeanCDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCDef[7], yerr = np.minimum(np.sqrt(rMeanCDef[7]), np.divide(rMeanCDef[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCDef[9], yerr = np.minimum(np.sqrt(rMeanCDef[9]), np.divide(rMeanCDef[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCDef[11], yerr = np.minimum(np.sqrt(rMeanCDef[11]), np.divide(rMeanCDef[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCDef[13], yerr = np.minimum(np.sqrt(rMeanCDef[13]), np.divide(rMeanCDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (large KLD, ordered)
plt.errorbar(ldaset, oMeanCDef[6], yerr = np.minimum(np.sqrt(oMeanCDef[6]), np.divide(oMeanCDef[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCDef[8], yerr = np.minimum(np.sqrt(oMeanCDef[8]), np.divide(oMeanCDef[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCDef[10], yerr = np.minimum(np.sqrt(oMeanCDef[10]), np.divide(oMeanCDef[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCDef[13], yerr = np.minimum(np.sqrt(oMeanCDef[13]), np.divide(oMeanCDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = default (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCDef[7], yerr = np.minimum(np.sqrt(oMeanCDef[7]), np.divide(oMeanCDef[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCDef[9], yerr = np.minimum(np.sqrt(oMeanCDef[9]), np.divide(oMeanCDef[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCDef[11], yerr = np.minimum(np.sqrt(oMeanCDef[11]), np.divide(oMeanCDef[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCDef[13], yerr = np.minimum(np.sqrt(oMeanCDef[13]), np.divide(oMeanCDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[4]} (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc_C_def.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCMid[0], yerr = np.minimum(np.sqrt(rMeanCMid[0]), np.divide(rMeanCMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCMid[2], yerr = np.minimum(np.sqrt(rMeanCMid[2]), np.divide(rMeanCMid[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCMid[4], yerr = np.minimum(np.sqrt(rMeanCMid[4]), np.divide(rMeanCMid[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCMid[12], yerr = np.minimum(np.sqrt(rMeanCMid[12]), np.divide(rMeanCMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCMid[1], yerr = np.minimum(np.sqrt(rMeanCMid[1]), np.divide(rMeanCMid[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCMid[3], yerr = np.minimum(np.sqrt(rMeanCMid[3]), np.divide(rMeanCMid[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCMid[5], yerr = np.minimum(np.sqrt(rMeanCMid[5]), np.divide(rMeanCMid[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCMid[12], yerr = np.minimum(np.sqrt(rMeanCMid[12]), np.divide(rMeanCMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (small KLD, ordered)
plt.errorbar(ldaset, oMeanCMid[0], yerr = np.minimum(np.sqrt(oMeanCMid[0]), np.divide(oMeanCMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCMid[2], yerr = np.minimum(np.sqrt(oMeanCMid[2]), np.divide(oMeanCMid[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCMid[4], yerr = np.minimum(np.sqrt(oMeanCMid[4]), np.divide(oMeanCMid[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCMid[12], yerr = np.minimum(np.sqrt(oMeanCMid[12]), np.divide(oMeanCMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCMid[1], yerr = np.minimum(np.sqrt(oMeanCMid[1]), np.divide(oMeanCMid[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCMid[3], yerr = np.minimum(np.sqrt(oMeanCMid[3]), np.divide(oMeanCMid[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCMid[5], yerr = np.minimum(np.sqrt(oMeanCMid[5]), np.divide(oMeanCMid[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCMid[12], yerr = np.minimum(np.sqrt(oMeanCMid[12]), np.divide(oMeanCMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCMid[6], yerr = np.minimum(np.sqrt(rMeanCMid[6]), np.divide(rMeanCMid[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCMid[8], yerr = np.minimum(np.sqrt(rMeanCMid[8]), np.divide(rMeanCMid[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCMid[10], yerr = np.minimum(np.sqrt(rMeanCMid[10]), np.divide(rMeanCMid[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCMid[13], yerr = np.minimum(np.sqrt(rMeanCMid[13]), np.divide(rMeanCMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCMid[7], yerr = np.minimum(np.sqrt(rMeanCMid[7]), np.divide(rMeanCMid[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCMid[9], yerr = np.minimum(np.sqrt(rMeanCMid[9]), np.divide(rMeanCMid[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCMid[11], yerr = np.minimum(np.sqrt(rMeanCMid[11]), np.divide(rMeanCMid[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCMid[13], yerr = np.minimum(np.sqrt(rMeanCMid[13]), np.divide(rMeanCMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (large KLD, ordered)
plt.errorbar(ldaset, oMeanCMid[6], yerr = np.minimum(np.sqrt(oMeanCMid[6]), np.divide(oMeanCMid[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCMid[8], yerr = np.minimum(np.sqrt(oMeanCMid[8]), np.divide(oMeanCMid[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCMid[10], yerr = np.minimum(np.sqrt(oMeanCMid[10]), np.divide(oMeanCMid[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCMid[13], yerr = np.minimum(np.sqrt(oMeanCMid[13]), np.divide(oMeanCMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = mid (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCMid[7], yerr = np.minimum(np.sqrt(oMeanCMid[7]), np.divide(oMeanCMid[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCMid[9], yerr = np.minimum(np.sqrt(oMeanCMid[9]), np.divide(oMeanCMid[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCMid[11], yerr = np.minimum(np.sqrt(oMeanCMid[11]), np.divide(oMeanCMid[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCMid[13], yerr = np.minimum(np.sqrt(oMeanCMid[13]), np.divide(oMeanCMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[7]} (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc_C_mid.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (small KLD, random +/-)
plt.errorbar(ldaset, rMeanCLarge[0], yerr = np.minimum(np.sqrt(rMeanCLarge[0]), np.divide(rMeanCLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCLarge[2], yerr = np.minimum(np.sqrt(rMeanCLarge[2]), np.divide(rMeanCLarge[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCLarge[4], yerr = np.minimum(np.sqrt(rMeanCLarge[4]), np.divide(rMeanCLarge[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCLarge[12], yerr = np.minimum(np.sqrt(rMeanCLarge[12]), np.divide(rMeanCLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (small KLD, random +/-)")
plt.savefig("Synth_C_est_small_rand_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCLarge[1], yerr = np.minimum(np.sqrt(rMeanCLarge[1]), np.divide(rMeanCLarge[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCLarge[3], yerr = np.minimum(np.sqrt(rMeanCLarge[3]), np.divide(rMeanCLarge[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCLarge[5], yerr = np.minimum(np.sqrt(rMeanCLarge[5]), np.divide(rMeanCLarge[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCLarge[12], yerr = np.minimum(np.sqrt(rMeanCLarge[12]), np.divide(rMeanCLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (small KLD, random +/-, mc)")
plt.savefig("Synth_C_est_small_rand_mc_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (small KLD, ordered)
plt.errorbar(ldaset, oMeanCLarge[0], yerr = np.minimum(np.sqrt(oMeanCLarge[0]), np.divide(oMeanCLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCLarge[2], yerr = np.minimum(np.sqrt(oMeanCLarge[2]), np.divide(oMeanCLarge[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCLarge[4], yerr = np.minimum(np.sqrt(oMeanCLarge[4]), np.divide(oMeanCLarge[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCLarge[12], yerr = np.minimum(np.sqrt(oMeanCLarge[12]), np.divide(oMeanCLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (small KLD, ordered)")
plt.savefig("Synth_C_est_small_ord_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCLarge[1], yerr = np.minimum(np.sqrt(oMeanCLarge[1]), np.divide(oMeanCLarge[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCLarge[3], yerr = np.minimum(np.sqrt(oMeanCLarge[3]), np.divide(oMeanCLarge[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCLarge[5], yerr = np.minimum(np.sqrt(oMeanCLarge[5]), np.divide(oMeanCLarge[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCLarge[12], yerr = np.minimum(np.sqrt(oMeanCLarge[12]), np.divide(oMeanCLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (small KLD, ordered, mc)")
plt.savefig("Synth_C_est_small_ord_mc_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (large KLD, random +/-)
plt.errorbar(ldaset, rMeanCLarge[6], yerr = np.minimum(np.sqrt(rMeanCLarge[6]), np.divide(rMeanCLarge[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanCLarge[8], yerr = np.minimum(np.sqrt(rMeanCLarge[8]), np.divide(rMeanCLarge[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanCLarge[10], yerr = np.minimum(np.sqrt(rMeanCLarge[10]), np.divide(rMeanCLarge[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanCLarge[13], yerr = np.minimum(np.sqrt(rMeanCLarge[13]), np.divide(rMeanCLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (large KLD, random +/-)")
plt.savefig("Synth_C_est_large_rand_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanCLarge[7], yerr = np.minimum(np.sqrt(rMeanCLarge[7]), np.divide(rMeanCLarge[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanCLarge[9], yerr = np.minimum(np.sqrt(rMeanCLarge[9]), np.divide(rMeanCLarge[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanCLarge[11], yerr = np.minimum(np.sqrt(rMeanCLarge[11]), np.divide(rMeanCLarge[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanCLarge[13], yerr = np.minimum(np.sqrt(rMeanCLarge[13]), np.divide(rMeanCLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (large KLD, random +/-, mc)")
plt.savefig("Synth_C_est_large_rand_mc_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (large KLD, ordered)
plt.errorbar(ldaset, oMeanCLarge[6], yerr = np.minimum(np.sqrt(oMeanCLarge[6]), np.divide(oMeanCLarge[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanCLarge[8], yerr = np.minimum(np.sqrt(oMeanCLarge[8]), np.divide(oMeanCLarge[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanCLarge[10], yerr = np.minimum(np.sqrt(oMeanCLarge[10]), np.divide(oMeanCLarge[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanCLarge[13], yerr = np.minimum(np.sqrt(oMeanCLarge[13]), np.divide(oMeanCLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (large KLD, ordered)")
plt.savefig("Synth_C_est_large_ord_C_large.png")
plt.clf()

# plot error of PRIEST-KLD when C = large (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanCLarge[7], yerr = np.minimum(np.sqrt(oMeanCLarge[7]), np.divide(oMeanCLarge[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanCLarge[9], yerr = np.minimum(np.sqrt(oMeanCLarge[9]), np.divide(oMeanCLarge[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanCLarge[11], yerr = np.minimum(np.sqrt(oMeanCLarge[11]), np.divide(oMeanCLarge[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanCLarge[13], yerr = np.minimum(np.sqrt(oMeanCLarge[13]), np.divide(oMeanCLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when C = {Cset[10]} (large KLD, ordered, mc)")
plt.savefig("Synth_C_est_large_ord_mc_C_large.png")
plt.clf()

# plot % of noise vs ground truth for each C (small KLD, random +/-)
plt.plot(Cset, rMeanPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, rMeanPerc[1], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(Cset, rMeanPerc[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, rMeanPerc[3], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(Cset, rMeanPerc[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, rMeanPerc[5], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (small KLD, random +/-)")
plt.savefig("Synth_C_perc_small_rand.png")
plt.clf()

# plot % of noise vs ground truth for each C (large KLD, random +/-)
plt.plot(Cset, rMeanPerc[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, rMeanPerc[7], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(Cset, rMeanPerc[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, rMeanPerc[9], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(Cset, rMeanPerc[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, rMeanPerc[11], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (large KLD, random +/-)")
plt.savefig("Synth_C_perc_large_rand.png")
plt.clf()

# plot % of noise vs ground truth for each C (small KLD, ordered)
plt.plot(Cset, oMeanPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, oMeanPerc[1], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(Cset, oMeanPerc[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, oMeanPerc[3], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(Cset, oMeanPerc[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, oMeanPerc[5], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (small KLD, ordered)")
plt.savefig("Synth_C_perc_small_ord.png")
plt.clf()

# plot % of noise vs ground truth for each C (large KLD, ordered)
plt.plot(Cset, oMeanPerc[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Cset, oMeanPerc[7], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(Cset, oMeanPerc[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Cset, oMeanPerc[9], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(Cset, oMeanPerc[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Cset, oMeanPerc[11], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of C")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (large KLD, ordered)")
plt.savefig("Synth_C_perc_large_ord.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")

