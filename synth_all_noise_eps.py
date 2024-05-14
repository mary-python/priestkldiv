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

for trial in range(8):
    print(f"Trial {trial}...")

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

    # 200 clients each with 125 points
    C = 200
    N = 125

    # parameters for the addition of Laplace and Gaussian noise
    epsset = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4])
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

    # load Laplace and Normal noise distributions, dependent on eps
    s1 = b1 / eps
    s2 = b2 / eps
    noiseL = dis.Laplace(loc = A, scale = s1)

    # numpy arrays
    rLda = 1
    ldaStep = 0.05
    L = int((rLda + ldaStep) / ldaStep)
    sEst = np.zeros((L, C))
    oEst = np.zeros((L, C))

    with alive_bar(C) as bar:
        for j in range(0, C):

            # even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                indices = torch.randperm(len(qPositiveRound))[:N]
                qClientSamp = qPositiveRound[indices]
            else:
                indices = torch.randperm(len(qNegativeRound))[:N]
                qClientSamp = qNegativeRound[indices]

            # each client gets N points in order from ordered pre-processed sample
            qOrdClientSamp = qOrderedRound[0][N*j : N*(j + 1)]

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

    # load Gaussian noise distribution for intermediate server
        if trial < 4:
            s3 = s2 * (np.sqrt(2) / R)
            midNoise = dis.Normal(loc = A, scale = s3)

    E = np.size(epsset)
    sMeanLda = np.zeros((L, E))
    oMeanLda = np.zeros((L, E))

    # compute mean of unbiased estimator across clients
    for l in np.arange(0, rLda + ldaStep, ldaStep):
        
        # option 3a: intermediate server adds noise term
        if trial < 4:
            sMeanLda[l] = np.mean(sEst, axis = 0) + midNoise.sample(sample_shape = (1,))
            oMeanLda[l] = np.mean(oEst, axis = 0) + midNoise.sample(sample_shape = (1,))
            
        # option 3b: server add noise term later
        else:
            sMeanLda[l] = np.mean(sEst, axis = 0)
            oMeanLda[l] = np.mean(oEst, axis = 0)

    # find lambda that produces minimum error
    sIndex = np.argmin(sMeanLda)
    oIndex = np.argmin(oMeanLda)

    # mean across clients for optimum lambda
    sMean = sMeanLda[sIndex]
    oMean = oMeanLda[oIndex]

    # numpy arrays
    sMeanA = np.zeros(E)
    oMeanA = np.zeros(E)
    EPS_COUNT = 0

    for eps in epsset:

        # option 3b: server adds noise term to final result
        if trial >= 4:
            sMeanA[EPS_COUNT] = (sMean + noiseL.sample(sample_shape = (1,)) - groundTruth)**2
            oMeanA[EPS_COUNT] = (oMean + noiseL.sample(sample_shape = (1,)) - groundTruth)**2

        # option 3a: intermediate server has already added noise term
        else:
            sMeanA[EPS_COUNT] = (sMean - groundTruth)**2
            oMeanA[EPS_COUNT] = (oMean - groundTruth)**2

        EPS_COUNT = EPS_COUNT + 1

    if trial % 8 == 0:
        sMean1 = sMeanA
        oMean1 = oMeanA

    if trial % 8 == 1:
        sMean2 = sMeanA
        oMean2 = oMeanA

    if trial % 8 == 2:
        sMean3 = sMeanA
        oMean3 = oMeanA

    if trial % 8 == 3:
        sMean4 = sMeanA
        oMean4 = oMeanA

    if trial % 8 == 4:
        sMean5 = sMeanA
        oMean5 = oMeanA
    
    if trial % 8 == 5:
        sMean6 = sMeanA
        oMean6 = oMeanA
    
    if trial % 8 == 6:
        sMean7 = sMeanA
        oMean7 = oMeanA

    if trial % 8 == 7:
        sMean8 = sMeanA
        oMean8 = oMeanA

# separate graphs for Small / Large KL divergence and end / mid noise to show trends
fig = plt.figure(figsize = (12.8, 9.6))

ax1 = plt.subplot(121)
ax1.plot(epsset, sMean1, label = "Small KLD + Gauss (samp)")
ax1.plot(epsset, oMean1, label = "Small KLD + Gauss (ord)")
ax1.plot(epsset, sMean2, label = "Small KLD + Gauss (samp) mc")
ax1.plot(epsset, oMean2, label = "Small KLD + Gauss (ord) mc")

ax1.set_title("Effect of epsilon on error of unbiased estimator")
ax1.set_xlabel("Value of epsilon")
ax1.set_ylabel("Error of unbiased estimator (mid noise)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(loc = "best")

ax2 = plt.subplot(122)
ax2.plot(epsset, sMean3, label = "Large KLD + Gauss (samp)")
ax2.plot(epsset, oMean3, label = "Large KLD + Gauss (ord)")
ax2.plot(epsset, sMean4, label = "Large KLD + Gauss (samp) mc")
ax2.plot(epsset, oMean4, label = "Large KLD + Gauss (ord) mc")

ax2.set_title("Effect of epsilon on error of unbiased estimator")
ax2.set_xlabel("Value of epsilon")
ax2.set_ylabel("Error of unbiased estimator (mid noise)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.legend(loc = "best")

ax3 = plt.subplot(221)
ax3.plot(epsset, sMean5, label = "Small KLD + Lap (samp)")
ax3.plot(epsset, oMean5, label = "Small KLD + Lap (ord)")
ax3.plot(epsset, sMean6, label = "Small KLD + Lap (samp) mc")
ax3.plot(epsset, oMean6, label = "Small KLD + Lap (ord) mc")

ax3.set_title("Effect of epsilon on error of unbiased estimator")
ax3.set_xlabel("Value of epsilon")
ax3.set_ylabel("Error of unbiased estimator (end noise)")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.legend(loc = "best")

ax4 = plt.subplot(222)
ax4.plot(epsset, sMean7, label = "Large KLD + Lap (samp)")
ax4.plot(epsset, oMean7, label = "Large KLD + Lap (ord)")
ax4.plot(epsset, sMean8, label = "Large KLD + Lap (samp) mc")
ax4.plot(epsset, oMean8, label = "Large KLD + Lap (ord) mc")

ax4.set_title("Effect of epsilon on error of unbiased estimator")
ax4.set_xlabel("Value of epsilon")
ax4.set_ylabel("Error of unbiased estimator (end noise)")
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.legend(loc = "best")

plt.tight_layout()
plt.savefig("Synth_all_noise_eps.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
