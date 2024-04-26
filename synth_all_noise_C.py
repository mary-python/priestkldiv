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

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial % 4 == 0 or trial % 4 == 1:
        p = dis.Laplace(loc = 0.1, scale = 1)

    # option 1b: distributions have large KL divergence
    else:
        p = dis.Laplace(loc = 1, scale = 1)

    q = dis.Normal(loc = 0, scale = 1)

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
    EPS = 0.1
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
    s1 = b1 / EPS
    s2 = b2 / EPS
    noiseL = dis.Laplace(loc = A, scale = s1)
    noiseN = dis.Normal(loc = A, scale = s2)

    # numpy arrays
    rLda = 1
    ldaStep = 0.05
    L = int((rLda + ldaStep) / ldaStep)
    CS = np.size(Cset)
    sMeanL = np.zeros(CS)
    sMeanN = np.zeros(CS)
    oMeanL = np.zeros(CS)
    oMeanN = np.zeros(CS)
    C_COUNT = 0

    print("Evaluating KL Divergence estimator...\n")

    for C in Cset:

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
        if trial >= 4:
            s3 = s2 * (np.sqrt(2) / R)
            midNoise = dis.Normal(loc = A, scale = s3)

        sMeanLda = np.zeros((L, CS))
        oMeanLda = np.zeros((L, CS))

        # compute mean of unbiased estimator across clients
        for l in range(0, L):

            # option 3a: intermediate server adds noise term
            if trial >= 4:
                sMeanLda[l] = np.mean(sEst[l]) + midNoise.sample(sample_shape = (1,))
                oMeanLda[l] = np.mean(oEst[l]) + midNoise.sample(sample_shape = (1,))
            
            # option 3b: server add noise term later
            else:
                sMeanLda[l] = np.mean(sEst[l])
                oMeanLda[l] = np.mean(oEst[l])

        # find lambda that produces minimum error
        sIndex = np.argmin(sMeanLda)
        oIndex = np.argmin(oMeanLda)

        # mean across clients for optimum lambda
        sMean = sMeanLda[sIndex]
        oMean = oMeanLda[oIndex]

        # option 3b: server adds noise term to final result
        if trial < 4:
            sMeanL[C_COUNT] = sMean + noiseL.sample(sample_shape = (1,))
            sMeanN[C_COUNT] = sMean + noiseN.sample(sample_shape = (1,))
            oMeanL[C_COUNT] = oMean + noiseL.sample(sample_shape = (1,))
            oMeanN[C_COUNT] = oMean + noiseN.sample(sample_shape = (1,))

        # option 3a: intermediate server has already added noise term
        else:
            sMeanL[C_COUNT] = sMean
            sMeanN[C_COUNT] = sMean
            oMeanL[C_COUNT] = oMean
            oMeanN[C_COUNT] = oMean

        C_COUNT = C_COUNT + 1

    if trial % 8 == 0:
        sMeanL1 = sMeanL
        sMeanN1 = sMeanN
        oMeanL1 = oMeanL
        oMeanN1 = oMeanL

    if trial % 8 == 1:
        sMeanL2 = sMeanL
        sMeanN2 = sMeanN
        oMeanL2 = oMeanL
        oMeanN2 = oMeanL

    if trial % 8 == 2:
        sMeanL3 = sMeanL
        sMeanN3 = sMeanN
        oMeanL3 = oMeanL
        oMeanN3 = oMeanL

    if trial % 8 == 3:
        sMeanL4 = sMeanL
        sMeanN4 = sMeanN
        oMeanL4 = oMeanL
        oMeanN4 = oMeanL
    
    if trial % 8 == 4:
        sMeanL5 = sMeanL
        sMeanN5 = sMeanN
        oMeanL5 = oMeanL
        oMeanN5 = oMeanL
    
    if trial % 8 == 5:
        sMeanL6 = sMeanL
        sMeanN6 = sMeanN
        oMeanL6 = oMeanL
        oMeanN6 = oMeanL
    
    if trial % 8 == 6:
        sMeanL7 = sMeanL
        sMeanN7 = sMeanN
        oMeanL7 = oMeanL
        oMeanN7 = oMeanL

    if trial % 8 == 7:
        sMeanL8 = sMeanL
        sMeanN8 = sMeanN
        oMeanL8 = oMeanL
        oMeanN8 = oMeanL

# separate graphs for Small / Large KL divergence and end / mid noise to show trends
fig = plt.figure(figsize = (12.8, 9.6))

ax1 = plt.subplot(121)
ax1.plot(Cset, sMeanL1, label = "Small KLD + Lap (samp)")
ax1.plot(Cset, sMeanN1, label = "Small KLD + Gauss (samp)")
ax1.plot(Cset, oMeanL1, label = "Small KLD + Lap (ord)")
ax1.plot(Cset, oMeanN1, label = "Small KLD + Gauss (ord)")
ax1.plot(Cset, sMeanL2, label = "Small KLD + Lap (samp) mc")
ax1.plot(Cset, sMeanN2, label = "Small KLD + Gauss (samp) mc")
ax1.plot(Cset, oMeanL2, label = "Small KLD + Lap (ord) mc")
ax1.plot(Cset, oMeanN2, label = "Small KLD + Gauss (ord) mc")

ax1.set_title("Effect of C on error of unbiased estimator")
ax1.set_xlabel("Value of C")
ax1.set_ylabel("Error of unbiased estimator (end noise)")
ax1.set_yscale("log")
ax1.legend(loc = "best")

ax2 = plt.subplot(122)
ax2.plot(Cset, sMeanL3, label = "Large KLD + Lap (samp)")
ax2.plot(Cset, sMeanN3, label = "Large KLD + Gauss (samp)")
ax2.plot(Cset, oMeanL3, label = "Large KLD + Lap (ord)")
ax2.plot(Cset, oMeanN3, label = "Large KLD + Gauss (ord)")
ax2.plot(Cset, sMeanL4, label = "Large KLD + Lap (samp) mc")
ax2.plot(Cset, sMeanN4, label = "Large KLD + Gauss (samp) mc")
ax2.plot(Cset, oMeanL4, label = "Large KLD + Lap (ord) mc")
ax2.plot(Cset, oMeanN4, label = "Large KLD + Gauss (ord) mc")

ax2.set_title("Effect of C on error of unbiased estimator")
ax2.set_xlabel("Value of C")
ax2.set_ylabel("Error of unbiased estimator (end noise)")
ax2.set_yscale("log")
ax2.legend(loc = "best")

ax3 = plt.subplot(221)
ax3.plot(Cset, sMeanL5, label = "Small KLD + Lap (samp)")
ax3.plot(Cset, sMeanN5, label = "Small KLD + Gauss (samp)")
ax3.plot(Cset, oMeanL5, label = "Small KLD + Lap (ord)")
ax3.plot(Cset, oMeanN5, label = "Small KLD + Gauss (ord)")
ax3.plot(Cset, sMeanL6, label = "Small KLD + Lap (samp) mc")
ax3.plot(Cset, sMeanN6, label = "Small KLD + Gauss (samp) mc")
ax3.plot(Cset, oMeanL6, label = "Small KLD + Lap (ord) mc")
ax3.plot(Cset, oMeanN6, label = "Small KLD + Gauss (ord) mc")

ax3.set_title("Effect of epsilon on error of unbiased estimator")
ax3.set_xlabel("Value of epsilon")
ax3.set_ylabel("Error of unbiased estimator (mid noise)")
ax3.set_yscale("log")
ax3.legend(loc = "best")

ax4 = plt.subplot(222)
ax4.plot(Cset, sMeanL7, label = "Large KLD + Lap (samp)")
ax4.plot(Cset, sMeanN7, label = "Large KLD + Gauss (samp)")
ax4.plot(Cset, oMeanL7, label = "Large KLD + Lap (ord)")
ax4.plot(Cset, oMeanN7, label = "Large KLD + Gauss (ord)")
ax4.plot(Cset, sMeanL8, label = "Large KLD + Lap (samp) mc")
ax4.plot(Cset, sMeanN8, label = "Large KLD + Gauss (samp) mc")
ax4.plot(Cset, oMeanL8, label = "Large KLD + Lap (ord) mc")
ax4.plot(Cset, oMeanN8, label = "Large KLD + Gauss (ord) mc")

ax4.set_title("Effect of epsilon on error of unbiased estimator")
ax4.set_xlabel("Value of epsilon")
ax4.set_ylabel("Error of unbiased estimator (mid noise)")
ax4.set_yscale("log")
ax4.legend(loc = "best")

plt.tight_layout()
plt.savefig("plot_all_noise_c.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
