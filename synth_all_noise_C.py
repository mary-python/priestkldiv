"""Modules provide various time-related functions as well as progress updates,
compute the natural logarithm of a number, parameterise probability distributions,
create static, animated, and interactive visualisations, and work with arrays."""
import time
from math import log
from alive_progress import alive_bar
import torch
import torch.distributions as dis
import matplotlib.pyplot as plt
import numpy as np

# initialising start time and seed for random sampling
startTime = time.perf_counter()
print("\nStarting...")
torch.manual_seed(12)

for trial in range(12):
    print(f"Trial {trial + 1}...")

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial % 4 == 0 or trial % 4 == 1:
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

    # sample different proportions of clients (1%-20%) each with 125 points
    Cset = np.array([40, 80, 120, 160, 200, 260, 320, 400, 480, 560, 620, 680])
    N = 125

    # parameters for the addition of Laplace and Gaussian noise
    EPS = 0.5
    DTA = 0.1
    A = 0
    R = 10

    # option 2a: baseline case
    if trial % 2 == 0:
        b1 = log(2)
    
    # option 2b: Monte-Carlo estimate
    else:
        b1 = 1 + log(2)

    b2 = 2*((log(1.25))/DTA)*b1

    # load Laplace and Gaussian noise distributions, dependent on eps
    s1 = b1 / EPS
    s2 = b2 / EPS
    lapNoise = dis.Laplace(loc = A, scale = s1)

    if trial < 8:
        s3 = s2 * (np.sqrt(2) / R)
        gaussNoise = dis.Normal(loc = A, scale = s3)

    # numpy arrays
    rLda = 1
    ldaStep = 0.05
    L = int((rLda + ldaStep) / ldaStep)
    CS = np.size(Cset)
    sMeanA = np.zeros(CS)
    oMeanA = np.zeros(CS)
    C_COUNT = 0

    for C in Cset:

        sEst = np.zeros((L, C))
        oEst = np.zeros((L, C))

        with alive_bar(C) as bar:
            for j in range(C):

                # even clients get positive values, odd clients get negative values
                if (j % 2) == 0:
                    indices = torch.randperm(len(qPositiveRound))[:N]
                    qClientSamp = qPositiveRound[indices]
                else:
                    indices = torch.randperm(len(qNegativeRound))[:N]
                    qClientSamp = qNegativeRound[indices]

                # each client gets N points in order from ordered pre-processed sample
                qOrdClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

                # option 3a: each client adds Gaussian noise term
                if trial < 4:
                    qOrdClientSamp = qOrdClientSamp + gaussNoise.sample(sample_shape = (1,))

                # compute ratio between unknown and known distributions
                sLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
                oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)
                LDA_COUNT = 0

                # explore lambdas in a range
                for lda in np.arange(0, rLda + ldaStep, ldaStep):

                    # compute k3 estimator
                    sRangeEst = (lda * (np.exp(sLogr) - 1)) - sLogr
                    oRangeEst = (lda * (np.exp(oLogr) - 1)) - oLogr

                    # share unbiased estimator with server
                    sEst[LDA_COUNT, j] = sRangeEst.mean()
                    oEst[LDA_COUNT, j] = oRangeEst.mean()
                    LDA_COUNT = LDA_COUNT + 1

                bar()

        sMeanLda = np.zeros((CS, L))
        oMeanLda = np.zeros((CS, L))

        # compute mean of unbiased estimator across clients
        for l in range(L):
            sMeanLda[l] = np.mean(sEst, axis = 1)
            oMeanLda[l] = np.mean(oEst, axis = 1)
        
            # option 3b: intermediate server adds Gaussian noise term
            if 4 <= trial < 8:
                sMeanLda[l] = sMeanLda[l] + gaussNoise.sample(sample_shape = (1,))
                oMeanLda[l] = oMeanLda[l] + gaussNoise.sample(sample_shape = (1,))

        # find lambda that produces minimum error
        sIndex = np.argmin(sMeanLda)
        oIndex = np.argmin(oMeanLda)

        # mean across clients for optimum lambda
        sMean = sMeanLda[sIndex]
        oMean = oMeanLda[oIndex]

        # option 3c: server adds Laplace noise term to final result
        if trial >= 8:
            sMeanA[C_COUNT] = (sMean + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2
            oMeanA[C_COUNT] = (oMean + lapNoise.sample(sample_shape = (1,)) - groundTruth)**2

        # option 3a/b: clients or intermediate server already added Gaussian noise term
        else:
            sMeanA[C_COUNT] = (sMean - groundTruth)**2
            oMeanA[C_COUNT] = (oMean - groundTruth)**2

        C_COUNT = C_COUNT + 1

    if trial % 12 == 0:
        sMean1 = sMeanA
        oMean1 = oMeanA

    if trial % 12 == 1:
        sMean2 = sMeanA
        oMean2 = oMeanA

    if trial % 12 == 2:
        sMean3 = sMeanA
        oMean3 = oMeanA

    if trial % 12 == 3:
        sMean4 = sMeanA
        oMean4 = oMeanA

    if trial % 12 == 4:
        sMean5 = sMeanA
        oMean5 = oMeanA
    
    if trial % 12 == 5:
        sMean6 = sMeanA
        oMean6 = oMeanA
    
    if trial % 12 == 6:
        sMean7 = sMeanA
        oMean7 = oMeanA

    if trial % 12 == 7:
        sMean8 = sMeanA
        oMean8 = oMeanA

    if trial % 12 == 8:
        sMean9 = sMeanA
        oMean9 = oMeanA
    
    if trial % 12 == 9:
        sMean10 = sMeanA
        oMean10 = oMeanA
    
    if trial % 12 == 10:
        sMean11 = sMeanA
        oMean11 = oMeanA

    if trial % 12 == 11:
        sMean12 = sMeanA
        oMean12 = oMeanA

# separate graphs for Small / Large KL divergence and end / mid noise to show trends
fig = plt.figure(figsize = (12.8, 14.4))

ax1 = plt.subplot(121)
ax1.plot(Cset, sMean1, label = "Small KLD + Gauss (samp)")
ax1.plot(Cset, oMean1, label = "Small KLD + Gauss (ord)")
ax1.plot(Cset, sMean2, label = "Small KLD + Gauss (samp) mc")
ax1.plot(Cset, oMean2, label = "Small KLD + Gauss (ord) mc")

ax1.set_title("Effect of C on error of unbiased estimator")
ax1.set_xlabel("Value of C")
ax1.set_ylabel("Error of unbiased estimator (start noise)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(loc = "best")

ax2 = plt.subplot(122)
ax2.plot(Cset, sMean3, label = "Large KLD + Gauss (samp)")
ax2.plot(Cset, oMean3, label = "Large KLD + Gauss (ord)")
ax2.plot(Cset, sMean4, label = "Large KLD + Gauss (samp) mc")
ax2.plot(Cset, oMean4, label = "Large KLD + Gauss (ord) mc")

ax2.set_title("Effect of C on error of unbiased estimator")
ax2.set_xlabel("Value of C")
ax2.set_ylabel("Error of unbiased estimator (start noise)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.legend(loc = "best")

ax3 = plt.subplot(221)
ax3.plot(Cset, sMean1, label = "Small KLD + Gauss (samp)")
ax3.plot(Cset, oMean1, label = "Small KLD + Gauss (ord)")
ax3.plot(Cset, sMean2, label = "Small KLD + Gauss (samp) mc")
ax3.plot(Cset, oMean2, label = "Small KLD + Gauss (ord) mc")

ax3.set_title("Effect of C on error of unbiased estimator")
ax3.set_xlabel("Value of C")
ax3.set_ylabel("Error of unbiased estimator (mid noise)")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.legend(loc = "best")

ax4 = plt.subplot(222)
ax4.plot(Cset, sMean3, label = "Large KLD + Gauss (samp)")
ax4.plot(Cset, oMean3, label = "Large KLD + Gauss (ord)")
ax4.plot(Cset, sMean4, label = "Large KLD + Gauss (samp) mc")
ax4.plot(Cset, oMean4, label = "Large KLD + Gauss (ord) mc")

ax4.set_title("Effect of C on error of unbiased estimator")
ax4.set_xlabel("Value of C")
ax4.set_ylabel("Error of unbiased estimator (mid noise)")
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.legend(loc = "best")

ax5 = plt.subplot(321)
ax5.plot(Cset, sMean5, label = "Small KLD + Lap (samp)")
ax5.plot(Cset, oMean5, label = "Small KLD + Lap (ord)")
ax5.plot(Cset, sMean6, label = "Small KLD + Lap (samp) mc")
ax5.plot(Cset, oMean6, label = "Small KLD + Lap (ord) mc")

ax5.set_title("Effect of C on error of unbiased estimator")
ax5.set_xlabel("Value of C")
ax5.set_ylabel("Error of unbiased estimator (end noise)")
ax5.set_xscale("log")
ax5.set_yscale("log")
ax5.legend(loc = "best")

ax6 = plt.subplot(322)
ax6.plot(Cset, sMean7, label = "Large KLD + Lap (samp)")
ax6.plot(Cset, oMean7, label = "Large KLD + Lap (ord)")
ax6.plot(Cset, sMean8, label = "Large KLD + Lap (samp) mc")
ax6.plot(Cset, oMean8, label = "Large KLD + Lap (ord) mc")

ax6.set_title("Effect of C on error of unbiased estimator")
ax6.set_xlabel("Value of C")
ax6.set_ylabel("Error of unbiased estimator (end noise)")
ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.legend(loc = "best")

plt.tight_layout()
plt.savefig("Synth_all_noise_C.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")

