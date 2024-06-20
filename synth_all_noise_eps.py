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

# lists of the values of epsilon and lambda, as well as the trials that will be explored
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist_small", "Dist_small_mc", "TAgg_small", "TAgg_small_mc", "Trusted_small", "Trusted_small_mc", "Dist_large",
            "Dist_large_mc", "TAgg_large", "TAgg_large_mc", "Trusted_large", "Trusted_large_mc", "NoAlgo_small", "NoAlgo_large"]
ES = len(epsset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates (random +/-)
rMeanEst = np.zeros((TS, ES))
rLdaOpt = np.zeros((TS, ES))
rLdaZero = np.zeros((TS, ES))
rLdaOne = np.zeros((TS, ES))
rLdaHalf = np.zeros((TS, ES))
rMeanPerc = np.zeros((TS, ES))
rMeanEpsSmall = np.zeros((TS, LS))
rMeanEpsDef = np.zeros((TS, LS))
rMeanEpsMid = np.zeros((TS, LS))
rMeanEpsLarge = np.zeros((TS, LS))

# to store statistics related to mean estimates (ordered)
oMeanEst = np.zeros((TS, ES))
oLdaOpt = np.zeros((TS, ES))
oLdaZero = np.zeros((TS, ES))
oLdaOne = np.zeros((TS, ES))
oLdaHalf = np.zeros((TS, ES))
oMeanPerc = np.zeros((TS, ES))
oMeanEpsSmall = np.zeros((TS, LS))
oMeanEpsDef = np.zeros((TS, LS))
oMeanEpsMid = np.zeros((TS, LS))
oMeanEpsLarge = np.zeros((TS, LS))

for trial in range(14):
    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    randfile = open(f"synth_eps_{trialset[trial]}_rand.txt", "w", encoding = 'utf-8')
    ordfile = open(f"synth_eps_{trialset[trial]}_ord.txt", "w", encoding = 'utf-8')

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

    # 200 clients each with 125 points
    C = 200
    N = 125

    # parameters for the addition of Laplace and Gaussian noise
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

    EPS_COUNT = 0

    for eps in epsset:
        print(f"Trial {trial + 1}: epsilon = {eps}...")

        if trial < 12:

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

            # option 3a: "Dist" (each client adds Gaussian noise term)
            if trial % 3 == 0 and trial != 12:
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
            
            # mean across lambdas for eps = 0.05 (small)
            if EPS_COUNT == 0:
                rMeanEpsSmall[trial, l] = rMeanLda[l]
                oMeanEpsSmall[trial, l] = oMeanLda[l]

            # eps = 0.5 (default)
            if EPS_COUNT == 3:
                rMeanEpsDef[trial, l] = rMeanLda[l]
                oMeanEpsDef[trial, l] = oMeanLda[l]

            # eps = 1.5 (mid)
            if EPS_COUNT == 6:
                rMeanEpsMid[trial, l] = rMeanLda[l]
                oMeanEpsMid[trial, l] = oMeanLda[l]

            # eps = 3 (large)
            if EPS_COUNT == 10:
                rMeanEpsLarge[trial, l] = rMeanLda[l]
                oMeanEpsLarge[trial, l] = oMeanLda[l]

        # find lambda that produces minimum error
        rLdaIndex = np.argmin(rMeanLda)
        oLdaIndex = np.argmin(oMeanLda)

        rMinMeanError = rMeanLda[rLdaIndex]
        oMinMeanError = oMeanLda[oLdaIndex]

        # mean across clients for optimum lambda
        rMeanEst[trial, EPS_COUNT] = rMinMeanError
        oMeanEst[trial, EPS_COUNT] = oMinMeanError

        # optimum lambda
        ldaStep = 0.05
        rLdaOpt[trial, EPS_COUNT] = rLdaIndex * ldaStep
        oLdaOpt[trial, EPS_COUNT] = oLdaIndex * ldaStep

        # lambda = 0
        rLdaZero[trial, EPS_COUNT] = rMeanLda[0]
        oLdaZero[trial, EPS_COUNT] = oMeanLda[0]

        # lambda = 1
        rLdaOne[trial, EPS_COUNT] = rMeanLda[LS-1]
        oLdaOne[trial, EPS_COUNT] = oMeanLda[LS-1]

        # lambda = 0.5
        rLdaHalf[trial, EPS_COUNT] = rMeanLda[(LS-1)/2]
        oLdaHalf[trial, EPS_COUNT] = oMeanLda[(LS-1)/2]

        # option 3c: "Trusted" (server adds Laplace noise term to final result)
        if trial % 3 == 2:
            rMeanNoise = lapNoise.sample(sample_shape = (1,))
            oMeanNoise = lapNoise.sample(sample_shape = (1,))

            # define error = squared difference between estimator and ground truth
            rMeanEst[trial, EPS_COUNT] = (rMeanEst[trial, EPS_COUNT] + rMeanNoise - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] + oMeanNoise - groundTruth)**2

        # clients or intermediate server already added Gaussian noise term
        else:
            rMeanEst[trial, EPS_COUNT] = (rMeanEst[trial, EPS_COUNT] - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] - groundTruth)**2

        if eps == epsset[0]:
            randfile.write(f"SYNTHETIC Random +/-: Eps = {eps}\n")
            ordfile.write(f"SYNTHETIC Ordered: Eps = {eps}\n")
        else:
            randfile.write(f"\nEps = {eps}\n")
            ordfile.write(f"\nEps = {eps}\n")

        randfile.write(f"\nMean Error: {round(rMeanEst[trial, EPS_COUNT], 2)}\n")
        randfile.write(f"Optimal Lambda: {round(rLdaOpt[trial, EPS_COUNT], 2)}\n")
        randfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")

        # compute % of noise vs ground truth (random +/-)
        if trial % 3 == 0 and trial != 12:
            rMeanPerc[trial, EPS_COUNT] = float(abs(np.array(sum(rStartNoise)) / (np.array(sum(rStartNoise) + groundTruth))))*100
            randfile.write(f"Noise: {np.round(rMeanPerc[trial, EPS_COUNT], 2)}%\n")
        if trial % 3 == 1 and trial != 13:
            rMeanPerc[trial, EPS_COUNT] = abs((np.sum(rMeanLdaNoise)) / (np.sum(rMeanLdaNoise) + groundTruth))*100
            randfile.write(f"Noise: {round(rMeanPerc[trial, EPS_COUNT], 2)}%\n")
        if trial % 3 == 2:
            rMeanPerc[trial, EPS_COUNT] = float(abs(np.array(rMeanNoise) / (np.array(rMeanNoise + groundTruth))))*100
            randfile.write(f"Noise: {np.round(rMeanPerc[trial, EPS_COUNT], 2)}%\n")

        ordfile.write(f"\nMean Error: {round(oMeanEst[trial, EPS_COUNT], 2)}\n")
        ordfile.write(f"Optimal Lambda: {round(oLdaOpt[trial, EPS_COUNT], 2)}\n")
        ordfile.write(f"Ground Truth: {round(float(groundTruth), 2)}\n")

        # compute % of noise vs ground truth (ordered)
        if trial % 3 == 0 and trial != 12:
            oMeanPerc[trial, EPS_COUNT] = float(abs(np.array(sum(oStartNoise)) / (np.array(sum(oStartNoise) + groundTruth))))*100
            ordfile.write(f"Noise: {np.round(oMeanPerc[trial, EPS_COUNT], 2)}%\n")
        if trial % 3 == 1 and trial != 13:
            oMeanPerc[trial, EPS_COUNT] = abs((np.sum(oMeanLdaNoise)) / (np.sum(oMeanLdaNoise) + groundTruth))*100
            ordfile.write(f"Noise: {round(oMeanPerc[trial, EPS_COUNT], 2)}%\n")
        if trial % 3 == 2:
            oMeanPerc[trial, EPS_COUNT] = float(abs(np.array(oMeanNoise) / (np.array(oMeanNoise + groundTruth))))*100
            ordfile.write(f"Noise: {np.round(oMeanPerc[trial, EPS_COUNT], 2)}%\n")

        EPS_COUNT = EPS_COUNT + 1

# plot error of PRIEST-KLD for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rMeanEst[0], yerr = np.minimum(np.sqrt(rMeanEst[0]), np.divide(rMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEst[2], yerr = np.minimum(np.sqrt(rMeanEst[2]), np.divide(rMeanEst[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEst[4], yerr = np.minimum(np.sqrt(rMeanEst[4]), np.divide(rMeanEst[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rMeanEst[12], yerr = np.minimum(np.sqrt(rMeanEst[12]), np.divide(rMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (small KLD, random +/-)")
plt.savefig("Synth_eps_small_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (small KLD, random +/-, mc)
plt.errorbar(epsset, rMeanEst[1], yerr = np.minimum(np.sqrt(rMeanEst[1]), np.divide(rMeanEst[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rMeanEst[3], yerr = np.minimum(np.sqrt(rMeanEst[3]), np.divide(rMeanEst[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rMeanEst[5], yerr = np.minimum(np.sqrt(rMeanEst[5]), np.divide(rMeanEst[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rMeanEst[12], yerr = np.minimum(np.sqrt(rMeanEst[12]), np.divide(rMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_small_rand_mc.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oMeanEst[0], yerr = np.minimum(np.sqrt(oMeanEst[0]), np.divide(oMeanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEst[2], yerr = np.minimum(np.sqrt(oMeanEst[2]), np.divide(oMeanEst[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEst[4], yerr = np.minimum(np.sqrt(oMeanEst[4]), np.divide(oMeanEst[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oMeanEst[12], yerr = np.minimum(np.sqrt(oMeanEst[12]), np.divide(oMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (small KLD, ordered)")
plt.savefig("Synth_eps_small_ord.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (small KLD, ordered, mc)
plt.errorbar(epsset, oMeanEst[1], yerr = np.minimum(np.sqrt(oMeanEst[1]), np.divide(oMeanEst[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oMeanEst[3], yerr = np.minimum(np.sqrt(oMeanEst[3]), np.divide(oMeanEst[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oMeanEst[5], yerr = np.minimum(np.sqrt(oMeanEst[5]), np.divide(oMeanEst[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oMeanEst[12], yerr = np.minimum(np.sqrt(oMeanEst[12]), np.divide(oMeanEst[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (small KLD, ordered, mc)")
plt.savefig("Synth_eps_small_ord_mc.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rMeanEst[6], yerr = np.minimum(np.sqrt(rMeanEst[6]), np.divide(rMeanEst[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEst[8], yerr = np.minimum(np.sqrt(rMeanEst[8]), np.divide(rMeanEst[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEst[10], yerr = np.minimum(np.sqrt(rMeanEst[10]), np.divide(rMeanEst[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rMeanEst[13], yerr = np.minimum(np.sqrt(rMeanEst[13]), np.divide(rMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (large KLD, random +/-)")
plt.savefig("Synth_eps_large_rand.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD, random +/-, mc)
plt.errorbar(epsset, rMeanEst[7], yerr = np.minimum(np.sqrt(rMeanEst[7]), np.divide(rMeanEst[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rMeanEst[9], yerr = np.minimum(np.sqrt(rMeanEst[9]), np.divide(rMeanEst[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rMeanEst[11], yerr = np.minimum(np.sqrt(rMeanEst[11]), np.divide(rMeanEst[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rMeanEst[13], yerr = np.minimum(np.sqrt(rMeanEst[13]), np.divide(rMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_large_rand_mc.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oMeanEst[6], yerr = np.minimum(np.sqrt(oMeanEst[6]), np.divide(oMeanEst[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEst[8], yerr = np.minimum(np.sqrt(oMeanEst[8]), np.divide(oMeanEst[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEst[10], yerr = np.minimum(np.sqrt(oMeanEst[10]), np.divide(oMeanEst[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oMeanEst[13], yerr = np.minimum(np.sqrt(oMeanEst[13]), np.divide(oMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (large KLD, ordered)")
plt.savefig("Synth_eps_large_ord.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (large KLD, ordered, mc)
plt.errorbar(epsset, oMeanEst[7], yerr = np.minimum(np.sqrt(oMeanEst[7]), np.divide(oMeanEst[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oMeanEst[9], yerr = np.minimum(np.sqrt(oMeanEst[9]), np.divide(oMeanEst[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oMeanEst[11], yerr = np.minimum(np.sqrt(oMeanEst[11]), np.divide(oMeanEst[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oMeanEst[13], yerr = np.minimum(np.sqrt(oMeanEst[13]), np.divide(oMeanEst[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = "best")
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Epsilon vs error of PRIEST-KLD (large KLD, ordered, mc)")
plt.savefig("Synth_eps_large_ord_mc.png")
plt.clf()

# plot optimum lambda for each epsilon (small KLD, random +/-)
plt.plot(epsset, rLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rLdaOpt[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rLdaOpt[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, rLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (small KLD, random +/-)")
plt.savefig("Synth_eps_lda_opt_small_rand.png")
plt.clf()

# plot optimum lambda for each epsilon (small KLD, random +/-, mc)
plt.plot(epsset, rLdaOpt[1], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(epsset, rLdaOpt[3], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(epsset, rLdaOpt[5], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(epsset, rLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_opt_small_rand_mc.png")
plt.clf()

# plot optimum lambda for each epsilon (small KLD, ordered)
plt.plot(epsset, oLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oLdaOpt[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oLdaOpt[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, oLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (small KLD, ordered)")
plt.savefig("Synth_eps_lda_opt_small_ord.png")
plt.clf()

# plot optimum lambda for each epsilon (small KLD, ordered, mc)
plt.plot(epsset, oLdaOpt[1], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(epsset, oLdaOpt[3], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(epsset, oLdaOpt[5], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(epsset, oLdaOpt[12], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (small KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_opt_small_ord_mc.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, random +/-)
plt.plot(epsset, rLdaOpt[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rLdaOpt[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rLdaOpt[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, rLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (large KLD, random +/-)")
plt.savefig("Synth_eps_lda_opt_large_rand.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, random +/-, mc)
plt.plot(epsset, rLdaOpt[7], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(epsset, rLdaOpt[9], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(epsset, rLdaOpt[11], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(epsset, rLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_opt_large_rand_mc.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, ordered)
plt.plot(epsset, oLdaOpt[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oLdaOpt[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oLdaOpt[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, oLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (large KLD, ordered)")
plt.savefig("Synth_eps_lda_opt_large_ord.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, ordered, mc)
plt.plot(epsset, oLdaOpt[7], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(epsset, oLdaOpt[9], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(epsset, oLdaOpt[11], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(epsset, oLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (large KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_opt_large_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rLdaZero[0], yerr = np.minimum(np.sqrt(rLdaZero[0]), np.divide(rLdaZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rLdaZero[2], yerr = np.minimum(np.sqrt(rLdaZero[2]), np.divide(rLdaZero[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rLdaZero[4], yerr = np.minimum(np.sqrt(rLdaZero[4]), np.divide(rLdaZero[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rLdaZero[12], yerr = np.minimum(np.sqrt(rLdaZero[12]), np.divide(rLdaZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, random +/-)")
plt.savefig("Synth_eps_lda_zero_small_rand.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, random +/-, mc)
plt.errorbar(epsset, rLdaZero[1], yerr = np.minimum(np.sqrt(rLdaZero[1]), np.divide(rLdaZero[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rLdaZero[3], yerr = np.minimum(np.sqrt(rLdaZero[3]), np.divide(rLdaZero[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rLdaZero[5], yerr = np.minimum(np.sqrt(rLdaZero[5]), np.divide(rLdaZero[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rLdaZero[12], yerr = np.minimum(np.sqrt(rLdaZero[12]), np.divide(rLdaZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_zero_small_rand_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oLdaZero[0], yerr = np.minimum(np.sqrt(oLdaZero[0]), np.divide(oLdaZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oLdaZero[2], yerr = np.minimum(np.sqrt(oLdaZero[2]), np.divide(oLdaZero[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oLdaZero[4], yerr = np.minimum(np.sqrt(oLdaZero[4]), np.divide(oLdaZero[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oLdaZero[12], yerr = np.minimum(np.sqrt(oLdaZero[12]), np.divide(oLdaZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, ordered)")
plt.savefig("Synth_eps_lda_zero_small_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, ordered, mc)
plt.errorbar(epsset, oLdaZero[1], yerr = np.minimum(np.sqrt(oLdaZero[1]), np.divide(oLdaZero[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oLdaZero[3], yerr = np.minimum(np.sqrt(oLdaZero[3]), np.divide(oLdaZero[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oLdaZero[5], yerr = np.minimum(np.sqrt(oLdaZero[5]), np.divide(oLdaZero[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oLdaZero[12], yerr = np.minimum(np.sqrt(oLdaZero[12]), np.divide(oLdaZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_zero_small_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rLdaZero[6], yerr = np.minimum(np.sqrt(rLdaZero[6]), np.divide(rLdaZero[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rLdaZero[8], yerr = np.minimum(np.sqrt(rLdaZero[8]), np.divide(rLdaZero[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rLdaZero[10], yerr = np.minimum(np.sqrt(rLdaZero[10]), np.divide(rLdaZero[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rLdaZero[13], yerr = np.minimum(np.sqrt(rLdaZero[13]), np.divide(rLdaZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, random +/-)")
plt.savefig("Synth_eps_lda_zero_large_rand.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, random +/-, mc)
plt.errorbar(epsset, rLdaZero[7], yerr = np.minimum(np.sqrt(rLdaZero[7]), np.divide(rLdaZero[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rLdaZero[9], yerr = np.minimum(np.sqrt(rLdaZero[9]), np.divide(rLdaZero[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rLdaZero[11], yerr = np.minimum(np.sqrt(rLdaZero[11]), np.divide(rLdaZero[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rLdaZero[13], yerr = np.minimum(np.sqrt(rLdaZero[13]), np.divide(rLdaZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_zero_large_rand_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oLdaZero[6], yerr = np.minimum(np.sqrt(oLdaZero[6]), np.divide(oLdaZero[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oLdaZero[8], yerr = np.minimum(np.sqrt(oLdaZero[8]), np.divide(oLdaZero[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oLdaZero[10], yerr = np.minimum(np.sqrt(oLdaZero[10]), np.divide(oLdaZero[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oLdaZero[13], yerr = np.minimum(np.sqrt(oLdaZero[13]), np.divide(oLdaZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, ordered)")
plt.savefig("Synth_eps_lda_zero_large_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, ordered, mc)
plt.errorbar(epsset, oLdaZero[7], yerr = np.minimum(np.sqrt(oLdaZero[7]), np.divide(oLdaZero[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oLdaZero[9], yerr = np.minimum(np.sqrt(oLdaZero[9]), np.divide(oLdaZero[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oLdaZero[11], yerr = np.minimum(np.sqrt(oLdaZero[11]), np.divide(oLdaZero[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oLdaZero[13], yerr = np.minimum(np.sqrt(oLdaZero[13]), np.divide(oLdaZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_zero_large_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rLdaOne[0], yerr = np.minimum(np.sqrt(rLdaOne[0]), np.divide(rLdaOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rLdaOne[2], yerr = np.minimum(np.sqrt(rLdaOne[2]), np.divide(rLdaOne[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rLdaOne[4], yerr = np.minimum(np.sqrt(rLdaOne[4]), np.divide(rLdaOne[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rLdaOne[12], yerr = np.minimum(np.sqrt(rLdaOne[12]), np.divide(rLdaOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, random +/-)")
plt.savefig("Synth_eps_lda_one_small_rand.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, random +/-, mc)
plt.errorbar(epsset, rLdaOne[1], yerr = np.minimum(np.sqrt(rLdaOne[1]), np.divide(rLdaOne[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rLdaOne[3], yerr = np.minimum(np.sqrt(rLdaOne[3]), np.divide(rLdaOne[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rLdaOne[5], yerr = np.minimum(np.sqrt(rLdaOne[5]), np.divide(rLdaOne[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rLdaOne[12], yerr = np.minimum(np.sqrt(rLdaOne[12]), np.divide(rLdaOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_one_small_rand_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oLdaOne[0], yerr = np.minimum(np.sqrt(oLdaOne[0]), np.divide(oLdaOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oLdaOne[2], yerr = np.minimum(np.sqrt(oLdaOne[2]), np.divide(oLdaOne[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oLdaOne[4], yerr = np.minimum(np.sqrt(oLdaOne[4]), np.divide(oLdaOne[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oLdaOne[12], yerr = np.minimum(np.sqrt(oLdaOne[12]), np.divide(oLdaOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, ordered)")
plt.savefig("Synth_eps_lda_one_small_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, ordered, mc)
plt.errorbar(epsset, oLdaOne[1], yerr = np.minimum(np.sqrt(oLdaOne[1]), np.divide(oLdaOne[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oLdaOne[3], yerr = np.minimum(np.sqrt(oLdaOne[3]), np.divide(oLdaOne[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oLdaOne[5], yerr = np.minimum(np.sqrt(oLdaOne[5]), np.divide(oLdaOne[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oLdaOne[12], yerr = np.minimum(np.sqrt(oLdaOne[12]), np.divide(oLdaOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_one_small_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rLdaOne[6], yerr = np.minimum(np.sqrt(rLdaOne[6]), np.divide(rLdaOne[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rLdaOne[8], yerr = np.minimum(np.sqrt(rLdaOne[8]), np.divide(rLdaOne[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rLdaOne[10], yerr = np.minimum(np.sqrt(rLdaOne[10]), np.divide(rLdaOne[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rLdaOne[13], yerr = np.minimum(np.sqrt(rLdaOne[13]), np.divide(rLdaOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, random +/-)")
plt.savefig("Synth_eps_lda_one_large_rand.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, random +/-, mc)
plt.errorbar(epsset, rLdaOne[7], yerr = np.minimum(np.sqrt(rLdaOne[7]), np.divide(rLdaOne[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rLdaOne[9], yerr = np.minimum(np.sqrt(rLdaOne[9]), np.divide(rLdaOne[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rLdaOne[11], yerr = np.minimum(np.sqrt(rLdaOne[11]), np.divide(rLdaOne[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rLdaOne[13], yerr = np.minimum(np.sqrt(rLdaOne[13]), np.divide(rLdaOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_one_large_rand_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oLdaOne[6], yerr = np.minimum(np.sqrt(oLdaOne[6]), np.divide(oLdaOne[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oLdaOne[8], yerr = np.minimum(np.sqrt(oLdaOne[8]), np.divide(oLdaOne[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oLdaOne[10], yerr = np.minimum(np.sqrt(oLdaOne[10]), np.divide(oLdaOne[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oLdaOne[13], yerr = np.minimum(np.sqrt(oLdaOne[13]), np.divide(oLdaOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, ordered)")
plt.savefig("Synth_eps_lda_one_large_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, ordered, mc)
plt.errorbar(epsset, oLdaOne[7], yerr = np.minimum(np.sqrt(oLdaOne[7]), np.divide(oLdaOne[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oLdaOne[9], yerr = np.minimum(np.sqrt(oLdaOne[9]), np.divide(oLdaOne[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oLdaOne[11], yerr = np.minimum(np.sqrt(oLdaOne[11]), np.divide(oLdaOne[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oLdaOne[13], yerr = np.minimum(np.sqrt(oLdaOne[13]), np.divide(oLdaOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_one_large_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rLdaHalf[0], yerr = np.minimum(np.sqrt(rLdaHalf[0]), np.divide(rLdaHalf[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rLdaHalf[2], yerr = np.minimum(np.sqrt(rLdaHalf[2]), np.divide(rLdaHalf[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rLdaHalf[4], yerr = np.minimum(np.sqrt(rLdaHalf[4]), np.divide(rLdaHalf[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rLdaHalf[12], yerr = np.minimum(np.sqrt(rLdaHalf[12]), np.divide(rLdaHalf[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (small KLD, random +/-)")
plt.savefig("Synth_eps_lda_half_small_rand.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (small KLD, random +/-, mc)
plt.errorbar(epsset, rLdaHalf[1], yerr = np.minimum(np.sqrt(rLdaHalf[1]), np.divide(rLdaHalf[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rLdaHalf[3], yerr = np.minimum(np.sqrt(rLdaHalf[3]), np.divide(rLdaHalf[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rLdaHalf[5], yerr = np.minimum(np.sqrt(rLdaHalf[5]), np.divide(rLdaHalf[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rLdaHalf[12], yerr = np.minimum(np.sqrt(rLdaHalf[12]), np.divide(rLdaHalf[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_half_small_rand_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oLdaHalf[0], yerr = np.minimum(np.sqrt(oLdaHalf[0]), np.divide(oLdaHalf[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oLdaHalf[2], yerr = np.minimum(np.sqrt(oLdaHalf[2]), np.divide(oLdaHalf[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oLdaHalf[4], yerr = np.minimum(np.sqrt(oLdaHalf[4]), np.divide(oLdaHalf[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oLdaHalf[12], yerr = np.minimum(np.sqrt(oLdaHalf[12]), np.divide(oLdaHalf[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (small KLD, ordered)")
plt.savefig("Synth_eps_lda_half_small_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (small KLD, ordered, mc)
plt.errorbar(epsset, oLdaHalf[1], yerr = np.minimum(np.sqrt(oLdaHalf[1]), np.divide(oLdaHalf[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oLdaHalf[3], yerr = np.minimum(np.sqrt(oLdaHalf[3]), np.divide(oLdaHalf[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oLdaHalf[5], yerr = np.minimum(np.sqrt(oLdaHalf[5]), np.divide(oLdaHalf[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oLdaHalf[12], yerr = np.minimum(np.sqrt(oLdaHalf[12]), np.divide(oLdaHalf[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (small KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_half_small_ord_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rLdaHalf[6], yerr = np.minimum(np.sqrt(rLdaHalf[6]), np.divide(rLdaHalf[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rLdaHalf[8], yerr = np.minimum(np.sqrt(rLdaHalf[8]), np.divide(rLdaHalf[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rLdaHalf[10], yerr = np.minimum(np.sqrt(rLdaHalf[10]), np.divide(rLdaHalf[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rLdaHalf[13], yerr = np.minimum(np.sqrt(rLdaHalf[13]), np.divide(rLdaHalf[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (large KLD, random +/-)")
plt.savefig("Synth_eps_lda_half_large_rand.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (large KLD, random +/-, mc)
plt.errorbar(epsset, rLdaHalf[7], yerr = np.minimum(np.sqrt(rLdaHalf[7]), np.divide(rLdaHalf[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rLdaHalf[9], yerr = np.minimum(np.sqrt(rLdaHalf[9]), np.divide(rLdaHalf[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rLdaHalf[11], yerr = np.minimum(np.sqrt(rLdaHalf[11]), np.divide(rLdaHalf[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rLdaHalf[13], yerr = np.minimum(np.sqrt(rLdaHalf[13]), np.divide(rLdaHalf[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_lda_half_large_rand_mc.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oLdaHalf[6], yerr = np.minimum(np.sqrt(oLdaHalf[6]), np.divide(oLdaHalf[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oLdaHalf[8], yerr = np.minimum(np.sqrt(oLdaHalf[8]), np.divide(oLdaHalf[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oLdaHalf[10], yerr = np.minimum(np.sqrt(oLdaHalf[10]), np.divide(oLdaHalf[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oLdaHalf[13], yerr = np.minimum(np.sqrt(oLdaHalf[13]), np.divide(oLdaHalf[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (large KLD, ordered)")
plt.savefig("Synth_eps_lda_half_large_ord.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0.5 for each epsilon (large KLD, ordered, mc)
plt.errorbar(epsset, oLdaHalf[7], yerr = np.minimum(np.sqrt(oLdaHalf[7]), np.divide(oLdaHalf[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oLdaHalf[9], yerr = np.minimum(np.sqrt(oLdaHalf[9]), np.divide(oLdaHalf[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oLdaHalf[11], yerr = np.minimum(np.sqrt(oLdaHalf[11]), np.divide(oLdaHalf[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oLdaHalf[13], yerr = np.minimum(np.sqrt(oLdaHalf[13]), np.divide(oLdaHalf[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0.5 (large KLD, ordered, mc)")
plt.savefig("Synth_eps_lda_half_large_ord_mc.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (small KLD, random +/-)
plt.plot(epsset, rMeanPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rMeanPerc[1], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(epsset, rMeanPerc[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rMeanPerc[3], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(epsset, rMeanPerc[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, rMeanPerc[5], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (small KLD, random +/-)")
plt.savefig("Synth_eps_perc_small_rand.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (large KLD, random +/-)
plt.plot(epsset, rMeanPerc[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, rMeanPerc[7], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(epsset, rMeanPerc[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, rMeanPerc[9], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(epsset, rMeanPerc[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, rMeanPerc[11], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (large KLD, random +/-)")
plt.savefig("Synth_eps_perc_large_rand.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (small KLD, ordered)
plt.plot(epsset, oMeanPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oMeanPerc[1], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(epsset, oMeanPerc[2], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oMeanPerc[3], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(epsset, oMeanPerc[4], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, oMeanPerc[5], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (small KLD, ordered)")
plt.savefig("Synth_eps_perc_small_ord.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (large KLD, ordered)
plt.plot(epsset, oMeanPerc[6], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, oMeanPerc[7], color = 'blueviolet', marker = 'x', label = "Dist mc")
plt.plot(epsset, oMeanPerc[8], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, oMeanPerc[9], color = 'lime', marker = 'x', label = "TAgg mc")
plt.plot(epsset, oMeanPerc[10], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, oMeanPerc[11], color = 'gold', marker = 'x', label = "Trusted mc")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.title("Noise (%) compared to ground truth (large KLD, ordered)")
plt.savefig("Synth_eps_perc_large_ord.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
