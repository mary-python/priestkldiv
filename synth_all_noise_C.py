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

# lists of the values of C and trials that will be run
Cset = [40, 80, 120, 160, 200, 260, 320, 400, 480, 560, 620, 680]
trialset = ["Dist_small", "Dist_small_mc", "TAgg_small", "TAgg_small_mc", "Trusted_small", "Trusted_small_mc", "Dist_large",
            "Dist_large_mc", "TAgg_large", "TAgg_large_mc", "Trusted_large", "Trusted_large_mc", "NoAlgo_small", "NoAlgo_large"]
CS = len(Cset)
TS = len(trialset)

# stores for mean of PRIEST-KLD, optimum and fixed lambdas, and % noise
rMeanEst = np.zeros((TS, CS))
rLdaOpt = np.zeros((TS, CS))
rLdaZero = np.zeros((TS, CS))
rLdaOne = np.zeros((TS, CS))
rLdaHalf = np.zeros((TS, CS))
rMeanPerc = np.zeros((TS, CS))

oMeanEst = np.zeros((TS, CS))
oLdaOpt = np.zeros((TS, CS))
oLdaZero = np.zeros((TS, CS))
oLdaOne = np.zeros((TS, CS))
oLdaHalf = np.zeros((TS, CS))
oMeanPerc = np.zeros((TS, CS))

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

    # numpy arrays
    rLda = 1
    ldaStep = 0.05
    L = int((rLda + ldaStep) / ldaStep)
    C_COUNT = 0

    for C in Cset:
        print(f"Trial {trial + 1}: C = {C}...")

        rEst = np.zeros((L, C))
        oEst = np.zeros((L, C))
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
            for lda in np.arange(0, rLda + ldaStep, ldaStep):

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

        rMeanLdaNoise = np.zeros(L)
        oMeanLdaNoise = np.zeros(L)
        
        # option 3b: "TAgg" (intermediate server adds Gaussian noise term)
        if trial % 3 == 1 and trial != 13:
            for l in range(L):
                rMeanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                oMeanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                rMeanLda[l] = rMeanLda[l] + rMeanLdaNoise[l]
                oMeanLda[l] = oMeanLda[l] + oMeanLdaNoise[l]

        # find lambda that produces minimum error
        rLdaIndex = np.argmin(rMeanLda)
        oLdaIndex = np.argmin(oMeanLda)

        rMinMeanError = rMeanLda[rLdaIndex]
        oMinMeanError = oMeanLda[oLdaIndex]

        # mean across clients for optimum lambda
        rMeanEst[trial, C_COUNT] = rMinMeanError
        oMeanEst[trial, C_COUNT] = oMinMeanError

        rLdaOpt[trial, C_COUNT] = rLdaIndex * ldaStep
        oLdaOpt[trial, C_COUNT] = oLdaIndex * ldaStep

        # lambda = 0
        rLdaZero[trial, C_COUNT] = rMeanLda[0]
        oLdaZero[trial, C_COUNT] = oMeanLda[0]

        # lambda = 1
        rLdaOne[trial, C_COUNT] = rMeanLda[L-1]
        oLdaOne[trial, C_COUNT] = oMeanLda[L-1]

        # lambda = 0.5
        rLdaHalf[trial, C_COUNT] = rMeanLda[(L-1)/2]
        oLdaHalf[trial, C_COUNT] = oMeanLda[(L-1)/2]

        # option 3c: "Trusted" (server adds Laplace noise term to final result)
        if trial % 3 == 2:
            rMeanNoise = lapNoise.sample(sample_shape = (1,))
            oMeanNoise = lapNoise.sample(sample_shape = (1,))

            # define error = squared difference between estimator and ground truth
            rMeanEst[trial, C_COUNT] = (rMeanEst[trial, C_COUNT] + rMeanNoise - groundTruth)**2
            oMeanEst[trial, C_COUNT] = (oMeanEst[trial, C_COUNT] + oMeanNoise - groundTruth)**2

        # clients or intermediate server already added Gaussian noise term
        else:
            rMeanEst[trial, C_COUNT] = (rMeanEst[trial, C_COUNT] - groundTruth)**2
            oMeanEst[trial, C_COUNT] = (oMeanEst[trial, C_COUNT] - groundTruth)**2

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
plt.savefig("Synth_C_small_rand.png")
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
plt.savefig("Synth_C_small_rand_mc.png")
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
plt.savefig("Synth_C_small_ord.png")
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
plt.savefig("Synth_C_small_ord_mc.png")
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
plt.savefig("Synth_C_large_rand.png")
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
plt.savefig("Synth_C_large_rand_mc.png")
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
plt.savefig("Synth_C_large_ord.png")
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
plt.savefig("Synth_C_large_ord_mc.png")
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
plt.savefig("Synth_C_lda_small_rand.png")
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
plt.savefig("Synth_C_lda_small_rand_mc.png")
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
plt.savefig("Synth_C_lda_small_ord.png")
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
plt.savefig("Synth_C_lda_small_ord_mc.png")
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
plt.savefig("Synth_C_lda_large_rand.png")
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
plt.savefig("Synth_C_lda_large_rand_mc.png")
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
plt.savefig("Synth_C_lda_large_ord.png")
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
plt.savefig("Synth_C_lda_large_ord_mc.png")
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

