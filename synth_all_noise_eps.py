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

# lists of the values of epsilon and trials that will be run
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
trialset = ["Dist_small", "Dist_small_mc", "TAgg_small", "TAgg_small_mc", "Trusted_small", "Trusted_small_mc", "Dist_large",
            "Dist_large_mc", "TAgg_large", "TAgg_large_mc", "Trusted_large", "Trusted_large_mc", "NoAlgo_small", "NoAlgo_large"]
ES = len(epsset)
TS = len(trialset)

# stores for mean of PRIEST-KLD and optimum lambda
rMeanEst = np.zeros((TS, ES))
rLdaOpt = np.zeros((TS, ES))
oMeanEst = np.zeros((TS, ES))
oLdaOpt = np.zeros((TS, ES))

for trial in range(14):
    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    statsfile = open(f"synth_eps_{trialset[trial]}.txt", "w", encoding = 'utf-8')

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
        rLda = 1
        ldaStep = 0.05
        L = int((rLda + ldaStep) / ldaStep)
        rEst = np.zeros((L, C))
        oEst = np.zeros((L, C))

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

            # option 3a: "Dist" (each client adds Gaussian noise term)
            if trial % 3 == 0 and trial != 12:
                rStartNoise = probGaussNoise.sample(sample_shape = (1,))
                oStartNoise = probGaussNoise.sample(sample_shape = (1,))

                if rStartNoise < 0:
                    qClientSamp = qClientSamp - rStartNoise
                else:
                    qClientSamp = qClientSamp + rStartNoise

                if oStartNoise < 0:
                    qClientSamp = qClientSamp - oStartNoise
                else:
                    qClientSamp = qClientSamp + oStartNoise

            # compute ratio between unknown and known distributions
            rLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
            oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)
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
        
        # option 3b: "TAgg" (intermediate server adds Gaussian noise term)
        if trial % 3 == 1 and trial != 13:
            for l in range(L):
                rMeanLda[l] = rMeanLda[l] + gaussNoise.sample(sample_shape = (1,))
                oMeanLda[l] = oMeanLda[l] + gaussNoise.sample(sample_shape = (1,))

        # find lambda that produces minimum error
        rLdaIndex = np.argmin(rMeanLda)
        oLdaIndex = np.argmin(oMeanLda)

        rMinMeanError = rMeanLda[rLdaIndex]
        oMinMeanError = oMeanLda[oLdaIndex]

        # mean across clients for optimum lambda
        rMeanEst[trial, EPS_COUNT] = rMinMeanError
        oMeanEst[trial, EPS_COUNT] = oMinMeanError

        rLdaOpt[trial, EPS_COUNT] = rLdaIndex * ldaStep
        oLdaOpt[trial, EPS_COUNT] = oLdaIndex * ldaStep

        # option 3c: "Trusted" (server adds Laplace noise term to final result)
        if trial % 3 == 2:
            rMeanEst[trial, EPS_COUNT] = (rMeanEst[trial, EPS_COUNT] + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2

        # clients or intermediate server already added Gaussian noise term
        else:
            rMeanEst[trial, EPS_COUNT] = (rMeanEst[trial, EPS_COUNT] - groundTruth)**2
            oMeanEst[trial, EPS_COUNT] = (oMeanEst[trial, EPS_COUNT] - groundTruth)**2

        statsfile.write(f"SYNTHETIC: Eps = {eps}\n")
        statsfile.write(f"Random +/-: Optimal Lambda {round(rLdaOpt[trial, EPS_COUNT], 2)} for Mean Error {round(rMeanEst[trial, EPS_COUNT], 2)}\n")
        statsfile.write(f"Ordered: Optimal Lambda {round(oLdaOpt[trial, EPS_COUNT], 2)} for Mean Error {round(oMeanEst[trial, EPS_COUNT], 2)}\n\n")

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
plt.savefig("Synth_eps_lda_small_rand.png")
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
plt.savefig("Synth_eps_lda_small_rand_mc.png")
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
plt.savefig("Synth_eps_lda_small_ord.png")
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
plt.savefig("Synth_eps_lda_small_ord_mc.png")
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
plt.savefig("Synth_eps_lda_large_rand.png")
plt.clf()

# plot optimum lambda for each epsilon (large KLD, random +/-, mc)
plt.plot(epsset, rLdaOpt[7], color = 'blueviolet', marker = 'x', label = "Dist")
plt.plot(epsset, rLdaOpt[9], color = 'lime', marker = 'x', label = "TAgg")
plt.plot(epsset, rLdaOpt[11], color = 'gold', marker = 'x', label = "Trusted")
plt.plot(epsset, rLdaOpt[13], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.title("Epsilon vs optimum lambda (large KLD, samp, mc)")
plt.savefig("Synth_eps_lda_large_rand_mc.png")
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
plt.savefig("Synth_eps_lda_large_ord.png")
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
plt.savefig("Synth_eps_lda_large_ord_mc.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
