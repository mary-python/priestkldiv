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

for trial in range(4):

    # p is unknown distribution, q is known
    # option 1a: distributions have small KL divergence
    if trial % 4 == 0 or trial % 4 == 1:
        p = dis.Laplace(loc = 0.1, scale = 1)
    
    # option 1b: distributions have large KL divergence
    else:
        p = dis.Laplace(loc = 1, scale = 1)

    q = dis.Normal(loc = 0, scale = 1)
    knownKLD = dis.kl_divergence(p, q)

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

    # 10-100K clients each with a few hundred points
    C = 10000
    N = 500
    M = (C / N) - 1

    # parameters for the addition of Laplace and Gaussian noise
    epsset = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4])
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

    # multi-dimensional numpy arrays
    E = np.size(epsset)
    rLda = 1
    ldaStep = 0.05
    L = int(rLda / ldaStep)
    sEst = np.zeros((C, E, L))
    oEst = np.zeros((C, E, L))
    print("Evaluating KL Divergence estimator...\n")

    with alive_bar(C*L) as bar:
        for j in range(0, C):

            # even clients get positive values, odd clients get negative values
            if (j % 2) == 0:
                indices = torch.randperm(len(qPositiveRound))[:N]
                qClientSamp = qPositiveRound[indices]
            else:
                indices = torch.randperm(len(qNegativeRound))[:N]
                qClientSamp = qNegativeRound[indices]

            qT = torch.numel(qClientSamp)

            # each client gets N points in order from ordered pre-processed sample
            # translated by M = (C / N) - 1 every time to stay below upper bound (C * N) - 1 for client C
            qOrdClientSamp = qOrderedRound[0][M*j : (M*j) + N]
            qOrderedT = torch.numel(qOrdClientSamp)
            EPS_COUNT = 0

            for eps in epsset:

                sNoiseLda = np.zeros(L)
                oNoiseLda = np.zeros(L)

                # load Gaussian noise distribution, dependent on eps and dta
                s1 = (b2 / eps) * (np.sqrt(2) / C)
                noise1 = dis.Normal(loc = A, scale = s1)
                
                sLogr = p.log_prob(qClientSamp) - q.log_prob(qClientSamp)
                oLogr = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp)

                # option 3a: intermediate server adds noise terms
                sNoise1 = p.log_prob(qClientSamp) - q.log_prob(qClientSamp) + noise1.sample(sample_shape = (qT,))
                oNoise1 = p.log_prob(qOrdClientSamp) - q.log_prob(qOrdClientSamp) + noise1.sample(sample_shape = (qOrderedT,))
                LDA_COUNT = 0

                # explore lambdas in a range
                for lda in range(0, rLda, ldaStep):

                    s2 = (b2 / eps) * ((lda * np.sqrt(2)) / C)
                    noise2 = dis.Normal(loc = A, scale = s2)

                    # compute k3 estimator
                    sNoise2 = sLogr.exp() + noise2.sample(sample_shape = (qT,))
                    oNoise2 = oLogr.exp() + noise2.sample(sample_shape = (qOrderedT,))

                    sNoiseLda = (sNoise2 - 1) - sNoise1
                    oNoiseLda = (oNoise2 - 1) - oNoise1

                    # compare with known KL divergence
                    sEst[j, EPS_COUNT, LDA_COUNT] = abs(sNoiseLda.mean() - knownKLD)
                    oEst[j, EPS_COUNT, LDA_COUNT] = abs(oNoiseLda.mean() - knownKLD)

                    LDA_COUNT = LDA_COUNT + 1

                EPS_COUNT = EPS_COUNT + 1
                bar()

    sMeanLda = np.zeros((L, E))
    oMeanLda = np.zeros((L, E))

    # compute mean of unbiased estimator across clients
    for l in range(0, rLda, ldaStep):
        sMeanLda[l] = np.mean(sEst, axis = (0, 1))
        oMeanLda[l] = np.mean(oEst, axis = (0, 1))

    # compute mean of unbiased estimator across epsilon
    sOpt = np.mean(sMeanLda, axis = 1)
    oOpt = np.mean(sMeanLda, axis = 1)

    # find lambda that produces minimum error
    sIndex = np.argmin(sOpt)
    oIndex = np.argmin(oOpt)

    sLda = ldaStep * sIndex
    oLda = ldaStep * oIndex

    # mean across clients for optimum lambda
    sMean = sMeanLda[sLda]
    oMean = sMeanLda[oLda]

    if trial % 4 == 0:
        sMean1 = sMean
        oMean1 = oMean
    
    if trial % 4 == 1:
        sMean2 = sMean
        oMean2 = oMean

    if trial % 4 == 2:
        sMean3 = sMean
        oMean3 = oMean

    if trial % 4 == 3:
        sMean4 = sMean
        oMean4 = oMean

# separate graphs for Small / Large KL divergence to show trends
fig = plt.figure(figsize = (12.8, 4.8))

ax1 = plt.subplot(121)
ax1.plot(epsset, sMean1, label = "Small KLD (samp)")
ax1.plot(epsset, oMean1, label = "Small KLD (ord)")
ax1.plot(epsset, sMean2, label = "Small KLD (samp) mc")
ax1.plot(epsset, oMean2, label = "Small KLD (ord) mc")

ax1.set_title("Effect of epsilon on error of unbiased estimator")
ax1.set_xlabel("Value of epsilon")
ax1.set_ylabel("Error of unbiased estimator (mid noise)")
ax1.set_yscale("log")
ax1.legend(loc = "best")

ax2 = plt.subplot(122)
ax2.plot(epsset, sMean3, label = "Large KLD (samp)")
ax2.plot(epsset, oMean3, label = "Large KLD (ord)")
ax2.plot(epsset, sMean4, label = "Large KLD (samp) mc")
ax2.plot(epsset, oMean4, label = "Large KLD (ord) mc")

ax2.set_title("Effect of epsilon on error of unbiased estimator")
ax2.set_xlabel("Value of epsilon")
ax2.set_ylabel("Error of unbiased estimator (mid noise)")
ax2.set_yscale("log")
ax2.legend(loc = "best")

plt.tight_layout()
plt.savefig("plot_mid_noise_eps.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
