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
rMeanEstZero = np.zeros((TS, ES))
rMeanEstOne = np.zeros((TS, ES))
rMeanPerc = np.zeros((TS, ES))
rMeanEpsSmall = np.zeros((TS, LS))
rMeanEpsDef = np.zeros((TS, LS))
rMeanEpsMid = np.zeros((TS, LS))
rMeanEpsLarge = np.zeros((TS, LS))

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
        rMeanEstZero[trial, EPS_COUNT] = rMeanLda[0]
        oMeanEstZero[trial, EPS_COUNT] = oMeanLda[0]

        # lambda = 1
        rMeanEstOne[trial, EPS_COUNT] = rMeanLda[LS-1]
        oMeanEstOne[trial, EPS_COUNT] = oMeanLda[LS-1]

        # option 3c: "Trusted" (server adds Laplace noise term to final result)
        if trial % 3 == 2:
            rMeanNoise = lapNoise.sample(sample_shape = (1,))
            oMeanNoise = lapNoise.sample(sample_shape = (1,))
            rMeanZeroNoise = lapNoise.sample(sample_shape = (1,))
            oMeanZeroNoise = lapNoise.sample(sample_shape = (1,))
            rMeanOneNoise = lapNoise.sample(sample_shape = (1,))
            oMeanOneNoise = lapNoise.sample(sample_shape = (1,))

            # define error = squared difference between estimator and ground truth
            rMeanEst[trial, EPS_COUNT] = (rMeanEst[trial, EPS_COUNT] + rMeanNoise - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] + oMeanNoise - groundTruth)**2

            # lambda = 0
            rMeanEstZero[trial, EPS_COUNT] = (rMeanEstZero[trial, EPS_COUNT] + rMeanZeroNoise - groundTruth)**2
            oMeanEstZero[trial, EPS_COUNT] = (oMeanEstZero[trial, EPS_COUNT] + oMeanZeroNoise - groundTruth)**2

            # lambda = 1
            rMeanEstOne[trial, EPS_COUNT] = (rMeanEstOne[trial, EPS_COUNT] + rMeanOneNoise - groundTruth)**2
            oMeanEstOne[trial, EPS_COUNT] = (oMeanEstOne[trial, EPS_COUNT] + oMeanOneNoise - groundTruth)**2
        

        # clients or intermediate server already added Gaussian noise term
        else:
            rMeanEst[trial, EPS_COUNT] = (rMeanEst[trial, EPS_COUNT] - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] - groundTruth)**2

            # lambda = 0
            rMeanEstZero[trial, EPS_COUNT] = (rMeanEstZero[trial, EPS_COUNT] - groundTruth)**2
            oMeanEstZero[trial, EPS_COUNT] = (oMeanEstZero[trial, EPS_COUNT] - groundTruth)**2

            # lambda = 1
            rMeanEstOne[trial, EPS_COUNT] = (rMeanEstOne[trial, EPS_COUNT] - groundTruth)**2
            oMeanEstOne[trial, EPS_COUNT] = (oMeanEstOne[trial, EPS_COUNT] - groundTruth)**2

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
plt.savefig("Synth_eps_est_small_rand.png")
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
plt.savefig("Synth_eps_est_small_rand_mc.png")
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
plt.savefig("Synth_eps_est_small_ord.png")
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
plt.savefig("Synth_eps_est_small_ord_mc.png")
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
plt.savefig("Synth_eps_est_large_rand.png")
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
plt.savefig("Synth_eps_est_large_rand_mc.png")
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
plt.savefig("Synth_eps_est_large_ord.png")
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
plt.savefig("Synth_eps_est_large_ord_mc.png")
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
plt.errorbar(epsset, rMeanEstZero[0], yerr = np.minimum(np.sqrt(rMeanEstZero[0]), np.divide(rMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstZero[2], yerr = np.minimum(np.sqrt(rMeanEstZero[2]), np.divide(rMeanEstZero[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstZero[4], yerr = np.minimum(np.sqrt(rMeanEstZero[4]), np.divide(rMeanEstZero[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rMeanEstZero[12], yerr = np.minimum(np.sqrt(rMeanEstZero[12]), np.divide(rMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, random +/-)")
plt.savefig("Synth_eps_est_small_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, random +/-, mc)
plt.errorbar(epsset, rMeanEstZero[1], yerr = np.minimum(np.sqrt(rMeanEstZero[1]), np.divide(rMeanEstZero[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rMeanEstZero[3], yerr = np.minimum(np.sqrt(rMeanEstZero[3]), np.divide(rMeanEstZero[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rMeanEstZero[5], yerr = np.minimum(np.sqrt(rMeanEstZero[5]), np.divide(rMeanEstZero[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rMeanEstZero[12], yerr = np.minimum(np.sqrt(rMeanEstZero[12]), np.divide(rMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_small_rand_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oMeanEstZero[0], yerr = np.minimum(np.sqrt(oMeanEstZero[0]), np.divide(oMeanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstZero[2], yerr = np.minimum(np.sqrt(oMeanEstZero[2]), np.divide(oMeanEstZero[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstZero[4], yerr = np.minimum(np.sqrt(oMeanEstZero[4]), np.divide(oMeanEstZero[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oMeanEstZero[12], yerr = np.minimum(np.sqrt(oMeanEstZero[12]), np.divide(oMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, ordered)")
plt.savefig("Synth_eps_est_small_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (small KLD, ordered, mc)
plt.errorbar(epsset, oMeanEstZero[1], yerr = np.minimum(np.sqrt(oMeanEstZero[1]), np.divide(oMeanEstZero[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oMeanEstZero[3], yerr = np.minimum(np.sqrt(oMeanEstZero[3]), np.divide(oMeanEstZero[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oMeanEstZero[5], yerr = np.minimum(np.sqrt(oMeanEstZero[5]), np.divide(oMeanEstZero[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oMeanEstZero[12], yerr = np.minimum(np.sqrt(oMeanEstZero[12]), np.divide(oMeanEstZero[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (small KLD, ordered, mc)")
plt.savefig("Synth_eps_est_small_ord_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rMeanEstZero[6], yerr = np.minimum(np.sqrt(rMeanEstZero[6]), np.divide(rMeanEstZero[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstZero[8], yerr = np.minimum(np.sqrt(rMeanEstZero[8]), np.divide(rMeanEstZero[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstZero[10], yerr = np.minimum(np.sqrt(rMeanEstZero[10]), np.divide(rMeanEstZero[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rMeanEstZero[13], yerr = np.minimum(np.sqrt(rMeanEstZero[13]), np.divide(rMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, random +/-)")
plt.savefig("Synth_eps_est_large_rand_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, random +/-, mc)
plt.errorbar(epsset, rMeanEstZero[7], yerr = np.minimum(np.sqrt(rMeanEstZero[7]), np.divide(rMeanEstZero[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rMeanEstZero[9], yerr = np.minimum(np.sqrt(rMeanEstZero[9]), np.divide(rMeanEstZero[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rMeanEstZero[11], yerr = np.minimum(np.sqrt(rMeanEstZero[11]), np.divide(rMeanEstZero[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rMeanEstZero[13], yerr = np.minimum(np.sqrt(rMeanEstZero[13]), np.divide(rMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_large_rand_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oMeanEstZero[6], yerr = np.minimum(np.sqrt(oMeanEstZero[6]), np.divide(oMeanEstZero[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstZero[8], yerr = np.minimum(np.sqrt(oMeanEstZero[8]), np.divide(oMeanEstZero[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstZero[10], yerr = np.minimum(np.sqrt(oMeanEstZero[10]), np.divide(oMeanEstZero[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oMeanEstZero[13], yerr = np.minimum(np.sqrt(oMeanEstZero[13]), np.divide(oMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, ordered)")
plt.savefig("Synth_eps_est_large_ord_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (large KLD, ordered, mc)
plt.errorbar(epsset, oMeanEstZero[7], yerr = np.minimum(np.sqrt(oMeanEstZero[7]), np.divide(oMeanEstZero[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oMeanEstZero[9], yerr = np.minimum(np.sqrt(oMeanEstZero[9]), np.divide(oMeanEstZero[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oMeanEstZero[11], yerr = np.minimum(np.sqrt(oMeanEstZero[11]), np.divide(oMeanEstZero[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oMeanEstZero[13], yerr = np.minimum(np.sqrt(oMeanEstZero[13]), np.divide(oMeanEstZero[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 0 (large KLD, ordered, mc)")
plt.savefig("Synth_eps_est_large_ord_mc_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, random +/-)
plt.errorbar(epsset, rMeanEstOne[0], yerr = np.minimum(np.sqrt(rMeanEstOne[0]), np.divide(rMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstOne[2], yerr = np.minimum(np.sqrt(rMeanEstOne[2]), np.divide(rMeanEstOne[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstOne[4], yerr = np.minimum(np.sqrt(rMeanEstOne[4]), np.divide(rMeanEstOne[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rMeanEstOne[12], yerr = np.minimum(np.sqrt(rMeanEstOne[12]), np.divide(rMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, random +/-)")
plt.savefig("Synth_eps_est_small_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, random +/-, mc)
plt.errorbar(epsset, rMeanEstOne[1], yerr = np.minimum(np.sqrt(rMeanEstOne[1]), np.divide(rMeanEstOne[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rMeanEstOne[3], yerr = np.minimum(np.sqrt(rMeanEstOne[3]), np.divide(rMeanEstOne[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rMeanEstOne[5], yerr = np.minimum(np.sqrt(rMeanEstOne[5]), np.divide(rMeanEstOne[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rMeanEstOne[12], yerr = np.minimum(np.sqrt(rMeanEstOne[12]), np.divide(rMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_small_rand_mc_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, ordered)
plt.errorbar(epsset, oMeanEstOne[0], yerr = np.minimum(np.sqrt(oMeanEstOne[0]), np.divide(oMeanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstOne[2], yerr = np.minimum(np.sqrt(oMeanEstOne[2]), np.divide(oMeanEstOne[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstOne[4], yerr = np.minimum(np.sqrt(oMeanEstOne[4]), np.divide(oMeanEstOne[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oMeanEstOne[12], yerr = np.minimum(np.sqrt(oMeanEstOne[12]), np.divide(oMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, ordered)")
plt.savefig("Synth_eps_est_small_ord_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (small KLD, ordered, mc)
plt.errorbar(epsset, oMeanEstOne[1], yerr = np.minimum(np.sqrt(oMeanEstOne[1]), np.divide(oMeanEstOne[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oMeanEstOne[3], yerr = np.minimum(np.sqrt(oMeanEstOne[3]), np.divide(oMeanEstOne[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oMeanEstOne[5], yerr = np.minimum(np.sqrt(oMeanEstOne[5]), np.divide(oMeanEstOne[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oMeanEstOne[12], yerr = np.minimum(np.sqrt(oMeanEstOne[12]), np.divide(oMeanEstOne[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (small KLD, ordered, mc)")
plt.savefig("Synth_eps_est_small_ord_mc_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, random +/-)
plt.errorbar(epsset, rMeanEstOne[6], yerr = np.minimum(np.sqrt(rMeanEstOne[6]), np.divide(rMeanEstOne[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, rMeanEstOne[8], yerr = np.minimum(np.sqrt(rMeanEstOne[8]), np.divide(rMeanEstOne[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, rMeanEstOne[10], yerr = np.minimum(np.sqrt(rMeanEstOne[10]), np.divide(rMeanEstOne[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, rMeanEstOne[13], yerr = np.minimum(np.sqrt(rMeanEstOne[13]), np.divide(rMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, random +/-)")
plt.savefig("Synth_eps_est_large_rand_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, random +/-, mc)
plt.errorbar(epsset, rMeanEstOne[7], yerr = np.minimum(np.sqrt(rMeanEstOne[7]), np.divide(rMeanEstOne[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, rMeanEstOne[9], yerr = np.minimum(np.sqrt(rMeanEstOne[9]), np.divide(rMeanEstOne[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, rMeanEstOne[11], yerr = np.minimum(np.sqrt(rMeanEstOne[11]), np.divide(rMeanEstOne[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, rMeanEstOne[13], yerr = np.minimum(np.sqrt(rMeanEstOne[13]), np.divide(rMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_large_rand_mc_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, ordered)
plt.errorbar(epsset, oMeanEstOne[6], yerr = np.minimum(np.sqrt(oMeanEstOne[6]), np.divide(oMeanEstOne[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, oMeanEstOne[8], yerr = np.minimum(np.sqrt(oMeanEstOne[8]), np.divide(oMeanEstOne[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, oMeanEstOne[10], yerr = np.minimum(np.sqrt(oMeanEstOne[10]), np.divide(oMeanEstOne[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, oMeanEstOne[13], yerr = np.minimum(np.sqrt(oMeanEstOne[13]), np.divide(oMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, ordered)")
plt.savefig("Synth_eps_est_large_ord_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (large KLD, ordered, mc)
plt.errorbar(epsset, oMeanEstOne[7], yerr = np.minimum(np.sqrt(oMeanEstOne[7]), np.divide(oMeanEstOne[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(epsset, oMeanEstOne[9], yerr = np.minimum(np.sqrt(oMeanEstOne[9]), np.divide(oMeanEstOne[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(epsset, oMeanEstOne[11], yerr = np.minimum(np.sqrt(oMeanEstOne[11]), np.divide(oMeanEstOne[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(epsset, oMeanEstOne[13], yerr = np.minimum(np.sqrt(oMeanEstOne[13]), np.divide(oMeanEstOne[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.title("Error of PRIEST-KLD when lambda = 1 (large KLD, ordered, mc)")
plt.savefig("Synth_eps_est_large_ord_mc_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsSmall[0], yerr = np.minimum(np.sqrt(rMeanEpsSmall[0]), np.divide(rMeanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsSmall[2], yerr = np.minimum(np.sqrt(rMeanEpsSmall[2]), np.divide(rMeanEpsSmall[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsSmall[4], yerr = np.minimum(np.sqrt(rMeanEpsSmall[4]), np.divide(rMeanEpsSmall[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsSmall[12], yerr = np.minimum(np.sqrt(rMeanEpsSmall[12]), np.divide(rMeanEpsSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (small KLD, random +/-)")
plt.savefig("Synth_eps_est_small_rand_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsSmall[1], yerr = np.minimum(np.sqrt(rMeanEpsSmall[1]), np.divide(rMeanEpsSmall[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsSmall[3], yerr = np.minimum(np.sqrt(rMeanEpsSmall[3]), np.divide(rMeanEpsSmall[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsSmall[5], yerr = np.minimum(np.sqrt(rMeanEpsSmall[5]), np.divide(rMeanEpsSmall[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsSmall[12], yerr = np.minimum(np.sqrt(rMeanEpsSmall[12]), np.divide(rMeanEpsSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_small_rand_mc_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsSmall[0], yerr = np.minimum(np.sqrt(oMeanEpsSmall[0]), np.divide(oMeanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsSmall[2], yerr = np.minimum(np.sqrt(oMeanEpsSmall[2]), np.divide(oMeanEpsSmall[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsSmall[4], yerr = np.minimum(np.sqrt(oMeanEpsSmall[4]), np.divide(oMeanEpsSmall[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsSmall[12], yerr = np.minimum(np.sqrt(oMeanEpsSmall[12]), np.divide(oMeanEpsSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (small KLD, ordered)")
plt.savefig("Synth_eps_est_small_ord_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsSmall[1], yerr = np.minimum(np.sqrt(oMeanEpsSmall[1]), np.divide(oMeanEpsSmall[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsSmall[3], yerr = np.minimum(np.sqrt(oMeanEpsSmall[3]), np.divide(oMeanEpsSmall[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsSmall[5], yerr = np.minimum(np.sqrt(oMeanEpsSmall[5]), np.divide(oMeanEpsSmall[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsSmall[12], yerr = np.minimum(np.sqrt(oMeanEpsSmall[12]), np.divide(oMeanEpsSmall[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (small KLD, ordered, mc)")
plt.savefig("Synth_eps_est_small_ord_mc_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsSmall[6], yerr = np.minimum(np.sqrt(rMeanEpsSmall[6]), np.divide(rMeanEpsSmall[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsSmall[8], yerr = np.minimum(np.sqrt(rMeanEpsSmall[8]), np.divide(rMeanEpsSmall[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsSmall[10], yerr = np.minimum(np.sqrt(rMeanEpsSmall[10]), np.divide(rMeanEpsSmall[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsSmall[13], yerr = np.minimum(np.sqrt(rMeanEpsSmall[13]), np.divide(rMeanEpsSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (large KLD, random +/-)")
plt.savefig("Synth_eps_est_large_rand_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsSmall[7], yerr = np.minimum(np.sqrt(rMeanEpsSmall[7]), np.divide(rMeanEpsSmall[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsSmall[9], yerr = np.minimum(np.sqrt(rMeanEpsSmall[9]), np.divide(rMeanEpsSmall[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsSmall[11], yerr = np.minimum(np.sqrt(rMeanEpsSmall[11]), np.divide(rMeanEpsSmall[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsSmall[13], yerr = np.minimum(np.sqrt(rMeanEpsSmall[13]), np.divide(rMeanEpsSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_large_rand_mc_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsSmall[6], yerr = np.minimum(np.sqrt(oMeanEpsSmall[6]), np.divide(oMeanEpsSmall[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsSmall[8], yerr = np.minimum(np.sqrt(oMeanEpsSmall[8]), np.divide(oMeanEpsSmall[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsSmall[10], yerr = np.minimum(np.sqrt(oMeanEpsSmall[10]), np.divide(oMeanEpsSmall[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsSmall[13], yerr = np.minimum(np.sqrt(oMeanEpsSmall[13]), np.divide(oMeanEpsSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (large KLD, ordered)")
plt.savefig("Synth_eps_est_large_ord_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = small (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsSmall[7], yerr = np.minimum(np.sqrt(oMeanEpsSmall[7]), np.divide(oMeanEpsSmall[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsSmall[9], yerr = np.minimum(np.sqrt(oMeanEpsSmall[9]), np.divide(oMeanEpsSmall[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsSmall[11], yerr = np.minimum(np.sqrt(oMeanEpsSmall[11]), np.divide(oMeanEpsSmall[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsSmall[13], yerr = np.minimum(np.sqrt(oMeanEpsSmall[13]), np.divide(oMeanEpsSmall[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[0]} (large KLD, ordered, mc)")
plt.savefig("Synth_eps_est_large_ord_mc_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsDef[0], yerr = np.minimum(np.sqrt(rMeanEpsDef[0]), np.divide(rMeanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsDef[2], yerr = np.minimum(np.sqrt(rMeanEpsDef[2]), np.divide(rMeanEpsDef[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsDef[4], yerr = np.minimum(np.sqrt(rMeanEpsDef[4]), np.divide(rMeanEpsDef[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsDef[12], yerr = np.minimum(np.sqrt(rMeanEpsDef[12]), np.divide(rMeanEpsDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (small KLD, random +/-)")
plt.savefig("Synth_eps_est_small_rand_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsDef[1], yerr = np.minimum(np.sqrt(rMeanEpsDef[1]), np.divide(rMeanEpsDef[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsDef[3], yerr = np.minimum(np.sqrt(rMeanEpsDef[3]), np.divide(rMeanEpsDef[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsDef[5], yerr = np.minimum(np.sqrt(rMeanEpsDef[5]), np.divide(rMeanEpsDef[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsDef[12], yerr = np.minimum(np.sqrt(rMeanEpsDef[12]), np.divide(rMeanEpsDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_small_rand_mc_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsDef[0], yerr = np.minimum(np.sqrt(oMeanEpsDef[0]), np.divide(oMeanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsDef[2], yerr = np.minimum(np.sqrt(oMeanEpsDef[2]), np.divide(oMeanEpsDef[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsDef[4], yerr = np.minimum(np.sqrt(oMeanEpsDef[4]), np.divide(oMeanEpsDef[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsDef[12], yerr = np.minimum(np.sqrt(oMeanEpsDef[12]), np.divide(oMeanEpsDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (small KLD, ordered)")
plt.savefig("Synth_eps_est_small_ord_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsDef[1], yerr = np.minimum(np.sqrt(oMeanEpsDef[1]), np.divide(oMeanEpsDef[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsDef[3], yerr = np.minimum(np.sqrt(oMeanEpsDef[3]), np.divide(oMeanEpsDef[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsDef[5], yerr = np.minimum(np.sqrt(oMeanEpsDef[5]), np.divide(oMeanEpsDef[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsDef[12], yerr = np.minimum(np.sqrt(oMeanEpsDef[12]), np.divide(oMeanEpsDef[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (small KLD, ordered, mc)")
plt.savefig("Synth_eps_est_small_ord_mc_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsDef[6], yerr = np.minimum(np.sqrt(rMeanEpsDef[6]), np.divide(rMeanEpsDef[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsDef[8], yerr = np.minimum(np.sqrt(rMeanEpsDef[8]), np.divide(rMeanEpsDef[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsDef[10], yerr = np.minimum(np.sqrt(rMeanEpsDef[10]), np.divide(rMeanEpsDef[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsDef[13], yerr = np.minimum(np.sqrt(rMeanEpsDef[13]), np.divide(rMeanEpsDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (large KLD, random +/-)")
plt.savefig("Synth_eps_est_large_rand_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsDef[7], yerr = np.minimum(np.sqrt(rMeanEpsDef[7]), np.divide(rMeanEpsDef[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsDef[9], yerr = np.minimum(np.sqrt(rMeanEpsDef[9]), np.divide(rMeanEpsDef[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsDef[11], yerr = np.minimum(np.sqrt(rMeanEpsDef[11]), np.divide(rMeanEpsDef[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsDef[13], yerr = np.minimum(np.sqrt(rMeanEpsDef[13]), np.divide(rMeanEpsDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_large_rand_mc_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsDef[6], yerr = np.minimum(np.sqrt(oMeanEpsDef[6]), np.divide(oMeanEpsDef[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsDef[8], yerr = np.minimum(np.sqrt(oMeanEpsDef[8]), np.divide(oMeanEpsDef[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsDef[10], yerr = np.minimum(np.sqrt(oMeanEpsDef[10]), np.divide(oMeanEpsDef[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsDef[13], yerr = np.minimum(np.sqrt(oMeanEpsDef[13]), np.divide(oMeanEpsDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (large KLD, ordered)")
plt.savefig("Synth_eps_est_large_ord_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = default (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsDef[7], yerr = np.minimum(np.sqrt(oMeanEpsDef[7]), np.divide(oMeanEpsDef[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsDef[9], yerr = np.minimum(np.sqrt(oMeanEpsDef[9]), np.divide(oMeanEpsDef[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsDef[11], yerr = np.minimum(np.sqrt(oMeanEpsDef[11]), np.divide(oMeanEpsDef[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsDef[13], yerr = np.minimum(np.sqrt(oMeanEpsDef[13]), np.divide(oMeanEpsDef[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[3]} (large KLD, ordered, mc)")
plt.savefig("Synth_eps_est_large_ord_mc_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsMid[0], yerr = np.minimum(np.sqrt(rMeanEpsMid[0]), np.divide(rMeanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsMid[2], yerr = np.minimum(np.sqrt(rMeanEpsMid[2]), np.divide(rMeanEpsMid[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsMid[4], yerr = np.minimum(np.sqrt(rMeanEpsMid[4]), np.divide(rMeanEpsMid[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsMid[12], yerr = np.minimum(np.sqrt(rMeanEpsMid[12]), np.divide(rMeanEpsMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (small KLD, random +/-)")
plt.savefig("Synth_eps_est_small_rand_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsMid[1], yerr = np.minimum(np.sqrt(rMeanEpsMid[1]), np.divide(rMeanEpsMid[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsMid[3], yerr = np.minimum(np.sqrt(rMeanEpsMid[3]), np.divide(rMeanEpsMid[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsMid[5], yerr = np.minimum(np.sqrt(rMeanEpsMid[5]), np.divide(rMeanEpsMid[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsMid[12], yerr = np.minimum(np.sqrt(rMeanEpsMid[12]), np.divide(rMeanEpsMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_small_rand_mc_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsMid[0], yerr = np.minimum(np.sqrt(oMeanEpsMid[0]), np.divide(oMeanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsMid[2], yerr = np.minimum(np.sqrt(oMeanEpsMid[2]), np.divide(oMeanEpsMid[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsMid[4], yerr = np.minimum(np.sqrt(oMeanEpsMid[4]), np.divide(oMeanEpsMid[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsMid[12], yerr = np.minimum(np.sqrt(oMeanEpsMid[12]), np.divide(oMeanEpsMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (small KLD, ordered)")
plt.savefig("Synth_eps_est_small_ord_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsMid[1], yerr = np.minimum(np.sqrt(oMeanEpsMid[1]), np.divide(oMeanEpsMid[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsMid[3], yerr = np.minimum(np.sqrt(oMeanEpsMid[3]), np.divide(oMeanEpsMid[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsMid[5], yerr = np.minimum(np.sqrt(oMeanEpsMid[5]), np.divide(oMeanEpsMid[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsMid[12], yerr = np.minimum(np.sqrt(oMeanEpsMid[12]), np.divide(oMeanEpsMid[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (small KLD, ordered, mc)")
plt.savefig("Synth_eps_est_small_ord_mc_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsMid[6], yerr = np.minimum(np.sqrt(rMeanEpsMid[6]), np.divide(rMeanEpsMid[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsMid[8], yerr = np.minimum(np.sqrt(rMeanEpsMid[8]), np.divide(rMeanEpsMid[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsMid[10], yerr = np.minimum(np.sqrt(rMeanEpsMid[10]), np.divide(rMeanEpsMid[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsMid[13], yerr = np.minimum(np.sqrt(rMeanEpsMid[13]), np.divide(rMeanEpsMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (large KLD, random +/-)")
plt.savefig("Synth_eps_est_large_rand_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsMid[7], yerr = np.minimum(np.sqrt(rMeanEpsMid[7]), np.divide(rMeanEpsMid[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsMid[9], yerr = np.minimum(np.sqrt(rMeanEpsMid[9]), np.divide(rMeanEpsMid[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsMid[11], yerr = np.minimum(np.sqrt(rMeanEpsMid[11]), np.divide(rMeanEpsMid[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsMid[13], yerr = np.minimum(np.sqrt(rMeanEpsMid[13]), np.divide(rMeanEpsMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_large_rand_mc_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsMid[6], yerr = np.minimum(np.sqrt(oMeanEpsMid[6]), np.divide(oMeanEpsMid[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsMid[8], yerr = np.minimum(np.sqrt(oMeanEpsMid[8]), np.divide(oMeanEpsMid[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsMid[10], yerr = np.minimum(np.sqrt(oMeanEpsMid[10]), np.divide(oMeanEpsMid[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsMid[13], yerr = np.minimum(np.sqrt(oMeanEpsMid[13]), np.divide(oMeanEpsMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (large KLD, ordered)")
plt.savefig("Synth_eps_est_large_ord_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = mid (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsMid[7], yerr = np.minimum(np.sqrt(oMeanEpsMid[7]), np.divide(oMeanEpsMid[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsMid[9], yerr = np.minimum(np.sqrt(oMeanEpsMid[9]), np.divide(oMeanEpsMid[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsMid[11], yerr = np.minimum(np.sqrt(oMeanEpsMid[11]), np.divide(oMeanEpsMid[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsMid[13], yerr = np.minimum(np.sqrt(oMeanEpsMid[13]), np.divide(oMeanEpsMid[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[6]} (large KLD, ordered, mc)")
plt.savefig("Synth_eps_est_large_ord_mc_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (small KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsLarge[0], yerr = np.minimum(np.sqrt(rMeanEpsLarge[0]), np.divide(rMeanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsLarge[2], yerr = np.minimum(np.sqrt(rMeanEpsLarge[2]), np.divide(rMeanEpsLarge[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsLarge[4], yerr = np.minimum(np.sqrt(rMeanEpsLarge[4]), np.divide(rMeanEpsLarge[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsLarge[12], yerr = np.minimum(np.sqrt(rMeanEpsLarge[12]), np.divide(rMeanEpsLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (small KLD, random +/-)")
plt.savefig("Synth_eps_est_small_rand_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (small KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsLarge[1], yerr = np.minimum(np.sqrt(rMeanEpsLarge[1]), np.divide(rMeanEpsLarge[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsLarge[3], yerr = np.minimum(np.sqrt(rMeanEpsLarge[3]), np.divide(rMeanEpsLarge[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsLarge[5], yerr = np.minimum(np.sqrt(rMeanEpsLarge[5]), np.divide(rMeanEpsLarge[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsLarge[12], yerr = np.minimum(np.sqrt(rMeanEpsLarge[12]), np.divide(rMeanEpsLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (small KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_small_rand_mc_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (small KLD, ordered)
plt.errorbar(ldaset, oMeanEpsLarge[0], yerr = np.minimum(np.sqrt(oMeanEpsLarge[0]), np.divide(oMeanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsLarge[2], yerr = np.minimum(np.sqrt(oMeanEpsLarge[2]), np.divide(oMeanEpsLarge[2], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsLarge[4], yerr = np.minimum(np.sqrt(oMeanEpsLarge[4]), np.divide(oMeanEpsLarge[4], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsLarge[12], yerr = np.minimum(np.sqrt(oMeanEpsLarge[12]), np.divide(oMeanEpsLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (small KLD, ordered)")
plt.savefig("Synth_eps_est_small_ord_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (small KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsLarge[1], yerr = np.minimum(np.sqrt(oMeanEpsLarge[1]), np.divide(oMeanEpsLarge[1], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsLarge[3], yerr = np.minimum(np.sqrt(oMeanEpsLarge[3]), np.divide(oMeanEpsLarge[3], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsLarge[5], yerr = np.minimum(np.sqrt(oMeanEpsLarge[5]), np.divide(oMeanEpsLarge[5], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsLarge[12], yerr = np.minimum(np.sqrt(oMeanEpsLarge[12]), np.divide(oMeanEpsLarge[12], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (small KLD, ordered, mc)")
plt.savefig("Synth_eps_est_small_ord_mc_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (large KLD, random +/-)
plt.errorbar(ldaset, rMeanEpsLarge[6], yerr = np.minimum(np.sqrt(rMeanEpsLarge[6]), np.divide(rMeanEpsLarge[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, rMeanEpsLarge[8], yerr = np.minimum(np.sqrt(rMeanEpsLarge[8]), np.divide(rMeanEpsLarge[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsLarge[10], yerr = np.minimum(np.sqrt(rMeanEpsLarge[10]), np.divide(rMeanEpsLarge[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsLarge[13], yerr = np.minimum(np.sqrt(rMeanEpsLarge[13]), np.divide(rMeanEpsLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (large KLD, random +/-)")
plt.savefig("Synth_eps_est_large_rand_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (large KLD, random +/-, mc)
plt.errorbar(ldaset, rMeanEpsLarge[7], yerr = np.minimum(np.sqrt(rMeanEpsLarge[7]), np.divide(rMeanEpsLarge[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, rMeanEpsLarge[9], yerr = np.minimum(np.sqrt(rMeanEpsLarge[9]), np.divide(rMeanEpsLarge[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, rMeanEpsLarge[11], yerr = np.minimum(np.sqrt(rMeanEpsLarge[11]), np.divide(rMeanEpsLarge[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, rMeanEpsLarge[13], yerr = np.minimum(np.sqrt(rMeanEpsLarge[13]), np.divide(rMeanEpsLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (large KLD, random +/-, mc)")
plt.savefig("Synth_eps_est_large_rand_mc_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (large KLD, ordered)
plt.errorbar(ldaset, oMeanEpsLarge[6], yerr = np.minimum(np.sqrt(oMeanEpsLarge[6]), np.divide(oMeanEpsLarge[6], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, oMeanEpsLarge[8], yerr = np.minimum(np.sqrt(oMeanEpsLarge[8]), np.divide(oMeanEpsLarge[8], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsLarge[10], yerr = np.minimum(np.sqrt(oMeanEpsLarge[10]), np.divide(oMeanEpsLarge[10], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsLarge[13], yerr = np.minimum(np.sqrt(oMeanEpsLarge[13]), np.divide(oMeanEpsLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (large KLD, ordered)")
plt.savefig("Synth_eps_est_large_ord_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = large (large KLD, ordered, mc)
plt.errorbar(ldaset, oMeanEpsLarge[7], yerr = np.minimum(np.sqrt(oMeanEpsLarge[7]), np.divide(oMeanEpsLarge[7], 2)), color = 'blueviolet', marker = 'x', label = "Dist")
plt.errorbar(ldaset, oMeanEpsLarge[9], yerr = np.minimum(np.sqrt(oMeanEpsLarge[9]), np.divide(oMeanEpsLarge[9], 2)), color = 'lime', marker = 'x', label = "TAgg")
plt.errorbar(ldaset, oMeanEpsLarge[11], yerr = np.minimum(np.sqrt(oMeanEpsLarge[11]), np.divide(oMeanEpsLarge[11], 2)), color = 'gold', marker = 'x', label = "Trusted")
plt.errorbar(ldaset, oMeanEpsLarge[13], yerr = np.minimum(np.sqrt(oMeanEpsLarge[13]), np.divide(oMeanEpsLarge[13], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.title(f"Error of PRIEST-KLD when epsilon = {epsset[10]} (large KLD, ordered, mc)")
plt.savefig("Synth_eps_est_large_ord_mc_eps_large.png")
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
