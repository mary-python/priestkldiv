"""Modules provide various time-related functions, compute the natural logarithm of a number,
create static, animated, and interactive visualisations, provide both a high- and low-level interface
to the HDF5 library, work with arrays, and carry out fast numerical computations in Python."""
import time
from math import log
import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
np.set_printoptions(suppress = True)
np.seterr(divide = 'ignore', invalid = 'ignore')
tf.random.set_seed(638)

# initialising start time and seed for random sampling
startTime = time.perf_counter()
print("\nStarting...")
np.random.seed(107642)

# from HDF5-FEMNIST by Xiao-Chenguang
# https://github.com/Xiao-Chenguang/HDF5-FEMNIST
# enables easy access and fast loading to the FEMNIST dataset from LEAF with the help of HDF5

# fetch HDF5 file from current directory
PATH = './data/write_all.hdf5'
file = h5py.File(PATH, 'r')

# create list storing images and labels of each writer
writers = sorted(file.keys())
numWriters = len(writers)
T = int(numWriters / 20)

# lists of the values of epsilon and trials that will be run
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
trialset = ["start_gauss", "start_gauss_mc", "mid_gauss", "mid_gauss_mc", "end_lap", "end_lap_mc", "no_privacy"]
ES = len(epsset)
TS = len(trialset)

# stores for mean of unbiased estimator and optimum lambda
meanEst = np.zeros((TS, ES))
meanLdaOpt = np.zeros((TS, ES))
meanUpper = np.zeros((TS, ES), dtype = bool)
meanLower = np.zeros((TS, ES), dtype = bool)

# for min pairs
minEst = np.zeros((TS, ES))
minLdaOpt = np.zeros((TS, ES))
minUpper = np.zeros((TS, ES), dtype = bool)
minLower = np.zeros((TS, ES), dtype = bool)

# for max pairs
maxEst = np.zeros((TS, ES))
maxLdaOpt = np.zeros((TS, ES))
maxUpper = np.zeros((TS, ES), dtype = bool)
maxLower = np.zeros((TS, ES), dtype = bool)

for trial in range(7):

    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    statsfile = open(f"femnist_{trialset[trial]}_noise.txt", "w", encoding = 'utf-8')
    EPS_FREQ = 0

    for eps in epsset:
        print(f"Trial {trial + 1}: epsilon = {eps}...")

        # randomly sample 5% of writers without replacement
        sampledWriters = np.random.choice(numWriters, T, replace = False)
        totalDigits = np.zeros(10, dtype = int)

        # compute the frequency of each digit
        for i in sampledWriters:
            tempDataset = file[writers[i]]

            for pic in range(len(tempDataset['labels'])):

                for freq in range(10):
                    if tempDataset['labels'][pic] == freq:
                        totalDigits[freq] = totalDigits[freq] + 1

        # create image store of appropriate dimensions for each digit
        zeroSet = np.ones((totalDigits[0], 4, 4), dtype = int)
        oneSet = np.ones((totalDigits[1], 4, 4), dtype = int)
        twoSet = np.ones((totalDigits[2], 4, 4), dtype = int)
        threeSet = np.ones((totalDigits[3], 4, 4), dtype = int)
        fourSet = np.ones((totalDigits[4], 4, 4), dtype = int)
        fiveSet = np.ones((totalDigits[5], 4, 4), dtype = int)
        sixSet = np.ones((totalDigits[6], 4, 4), dtype = int)
        sevenSet = np.ones((totalDigits[7], 4, 4), dtype = int)
        eightSet = np.ones((totalDigits[8], 4, 4), dtype = int)
        nineSet = np.ones((totalDigits[9], 4, 4), dtype = int)

        # to store condensed image and frequency of each digit
        smallPic = np.ones((4, 4), dtype = int)
        digFreq = np.zeros(10, dtype = int)

        def add_digit(dset):
            """Method to add digit to set corresponding to label."""
            dset[digFreq[label]] = smallPic

        for i in sampledWriters:

            tempDataset = file[writers[i]]
            PIC_FREQ = 0

            for pic in tempDataset['images']:

                # partition each image into 16 7x7 subimages
                for a in range(4):
                    for b in range(4):
                        subImage = pic[7*a : 7*(a + 1), 7*b : 7*(b + 1)]

                        # save rounded mean of each subimage into corresponding cell of smallpic
                        meanSubImage = np.mean(subImage)
                        if meanSubImage == 255:
                            smallPic[a, b] = 1
                        else:
                            smallPic[a, b] = 0

                label = tempDataset['labels'][PIC_FREQ]

                # split images according to label
                if label == 0:
                    add_digit(zeroSet)
                elif label == 1:
                    add_digit(oneSet)
                elif label == 2:
                    add_digit(twoSet)
                elif label == 3:
                    add_digit(threeSet)
                elif label == 4:
                    add_digit(fourSet)
                elif label == 5:
                    add_digit(fiveSet)
                elif label == 6:
                    add_digit(sixSet)
                elif label == 7:
                    add_digit(sevenSet)
                elif label == 8:
                    add_digit(eightSet)
                elif label == 9:
                    add_digit(nineSet)

                digFreq[label] = digFreq[label] + 1
                PIC_FREQ = PIC_FREQ + 1

        # store frequency of unique images corresponding to each digit
        sizeUSet = np.zeros(11)

        def unique_images(dg, dset):
            """Method to return unique images of set corresponding to digit."""
            uset = np.unique(dset, axis = 0)
            sizeUSet[dg] = len(uset)
            return uset

        uZeroSet = unique_images(0, zeroSet)
        uOneSet = unique_images(1, oneSet)
        uTwoSet = unique_images(2, twoSet)
        uThreeSet = unique_images(3, threeSet)
        uFourSet = unique_images(4, fourSet)
        uFiveSet = unique_images(5, fiveSet)
        uSixSet = unique_images(6, sixSet)
        uSevenSet = unique_images(7, sevenSet)
        uEightSet = unique_images(8, eightSet)
        uNineSet = unique_images(9, nineSet)

        # store frequency of unique images in total
        uTotalFreq = int(sum(sizeUSet))
        uTotalSet = np.ones((uTotalFreq, 4, 4), dtype = int)
        TOTAL_FREQ = 0

        def total_set(uset, tset, tfreq):
            """Method to add each of the unique images for each digit."""
            for im in uset:
                tset[tfreq] = im
                tfreq = tfreq + 1
            return tfreq

        TOTAL_FREQ = total_set(uZeroSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uOneSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uTwoSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uThreeSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uFourSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uFiveSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uSixSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uSevenSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uEightSet, uTotalSet, TOTAL_FREQ)
        TOTAL_FREQ = total_set(uNineSet, uTotalSet, TOTAL_FREQ)

        uTotalSet = unique_images(10, uTotalSet)

        # domain for each digit distribution is number of unique images
        U = len(uTotalSet)

        # store frequencies of unique images for each digit
        uImageSet = np.ones((10, U, 4, 4))
        uFreqSet = np.zeros((10, U))
        uProbsSet = np.zeros((10, U))
        T1 = 11*T # change this term so probabilities add up to 1

        # parameters for the addition of Laplace and Gaussian noise
        DTA = 0.1
        A = 0
        R = 90

        # option 1a: baseline case
        if trial % 2 == 0:
            b1 = log(2) / eps

        # option 1b: Monte Carlo estimate
        else:
            b1 = (1 + log(2)) / eps

        b2 = (2*((log(1.25))/DTA)*b1) / eps

        # load Gaussian noise distributions for clients and intermediate server
        if trial < 4:
            s = b2 * (np.sqrt(2) / R)
            gaussNoise = tfp.distributions.Normal(loc = A, scale = s)

        # smoothing parameter: 0.1 and 1 are too large
        ALPHA = 0.01

        def smoothed_prob(dset, dig, im, ufreq):
            """Method to compute frequencies of unique images and return smoothed probabilities."""
            where = np.where(np.all(im == dset, axis = (1, 2)))
            freq = len(where[0])
            uImageSet[dig, ufreq] = im
            uFreqSet[dig, ufreq] = int(freq)
            
            # option 2a: each client adds Gaussian noise term
            if trial == 0 or trial == 1:
                uProbsSet[dig, ufreq] = float((freq + ALPHA)/(T1 + (ALPHA*(digFreq[dig])))) + gaussNoise.sample(sample_shape = (1,))

            # option 2b/c: no noise until later
            else:
                uProbsSet[dig, ufreq] = float((freq + ALPHA)/(T1 + (ALPHA*(digFreq[dig]))))

        for D in range(0, 10):
            UNIQUE_FREQ = 0

            # store image and smoothed probability as well as frequency
            for image in uTotalSet:
                if D == 0:
                    smoothed_prob(zeroSet, 0, image, UNIQUE_FREQ)
                elif D == 1:
                    smoothed_prob(oneSet, 1, image, UNIQUE_FREQ)
                elif D == 2:
                    smoothed_prob(twoSet, 2, image, UNIQUE_FREQ)
                elif D == 3:
                    smoothed_prob(threeSet, 3, image, UNIQUE_FREQ)
                elif D == 4:
                    smoothed_prob(fourSet, 4, image, UNIQUE_FREQ)
                elif D == 5:
                    smoothed_prob(fiveSet, 5, image, UNIQUE_FREQ)
                elif D == 6:
                    smoothed_prob(sixSet, 6, image, UNIQUE_FREQ)
                elif D == 7:
                    smoothed_prob(sevenSet, 7, image, UNIQUE_FREQ)
                elif D == 8:
                    smoothed_prob(eightSet, 8, image, UNIQUE_FREQ)
                elif D == 9:
                    smoothed_prob(nineSet, 9, image, UNIQUE_FREQ)

                UNIQUE_FREQ = UNIQUE_FREQ + 1

        # for k3 estimator (Schulman) take a small sample of unique images
        E = 17

        # store images, frequencies and probabilities for this subset
        eImageSet = np.ones((10, E, 4, 4))
        eFreqSet = np.zeros((10, E))
        eProbsSet = np.zeros((10, E))
        eTotalFreq = np.zeros(10)

        uSampledSet = np.random.choice(U, E, replace = False)
        T2 = (11/3)*T*(E/U) # change this term so probabilities add up to 1

        # borrow data from corresponding indices of main image and frequency sets
        for D in range(0, 10):
            for i in range(E):
                eImageSet[D, i] = uImageSet[D, uSampledSet[i]]
                eFreqSet[D, i] = uFreqSet[D, uSampledSet[i]]
                eTotalFreq[D] = sum(eFreqSet[D])
                eProbsSet[D, i] = float((eFreqSet[D, i] + ALPHA)/(T2 + (ALPHA*(eTotalFreq[D]))))

        # stores for exact unknown distributions
        uDist = np.zeros((10, 10, U))
        nDist = np.zeros((10, 10, E))
        uList = []
        uCDList = []
        rList = []

        # for each comparison digit compute unknown distributions for all digits
        for C in range(0, 10):
            for D in range(0, 10):

                for i in range(0, U):
                    uDist[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

                # eliminate all zero values when digits are identical
                if sum(uDist[C, D]) != 0.0:
                    uList.append(sum(uDist[C, D]))
                    uCDList.append((C, D))

                for j in range(0, E):
                    nDist[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

                # compute ratio between exact unknown distributions
                ratio = abs(sum(nDist[C, D]) / sum(uDist[C, D]))

                # eliminate all divide by zero errors
                if ratio != 0.0 and sum(uDist[C, D]) != 0.0:
                    rList.append(ratio)

        # constants for lambda search
        rLda = 1
        ldaStep = 0.05
        L = int((rLda + ldaStep) / ldaStep)

        # store for unbiased estimator
        uEst = np.zeros((L, R))
        R_FREQ = 0

        for row in range(0, R):
            uLogr = np.log(rList[row])
            LDA_FREQ = 0

            # explore lambdas in a range
            for lda in np.arange(0, rLda + ldaStep, ldaStep):

                # compute k3 estimator
                uRangeEst = lda * (np.exp(uLogr) - 1) - uLogr

                # share unbiased estimator with server
                uEst[LDA_FREQ, R_FREQ] = uRangeEst
                LDA_FREQ = LDA_FREQ + 1
            
            R_FREQ = R_FREQ + 1
        
        # extract position and identity of max and min pairs
        minIndex = np.argmin(uList)
        maxIndex = np.argmax(uList)
        minPair = uCDList[minIndex]
        maxPair = uCDList[maxIndex]

        meanLda = np.zeros(L)
        minLda = np.zeros(L)
        maxLda = np.zeros(L)

        # compute mean error of unbiased estimator for each lambda
        for l in range(0, L):

            # option 2b: intermediate server adds Gaussian noise term
            if trial == 2 or trial == 3:
                meanLda[l] = np.mean(uEst[l]) + gaussNoise.sample(sample_shape = (1,))

                # extract error for max and min pairs
                minLda[l] = uEst[l, minIndex] + gaussNoise.sample(sample_shape = (1,))
                maxLda[l] = uEst[l, maxIndex] + gaussNoise.sample(sample_shape = (1,))

            # option 2a/c: noise already added at start or no noise until end
            else:
                meanLda[l] = np.mean(uEst[l])

                # extract error for max and min pairs
                minLda[l] = uEst[l, minIndex]
                maxLda[l] = uEst[l, maxIndex]

        # find lambda that produces minimum error
        meanLdaIndex = np.argmin(meanLda)
        minLdaIndex = np.argmin(minLda)
        maxLdaIndex = np.argmin(maxLda)

        meanMinError = meanLda[meanLdaIndex]
        minMinError = minLda[minLdaIndex]
        maxMinError = maxLda[maxLdaIndex]

        # mean across clients for optimum lambda
        meanEst[trial, EPS_FREQ] = meanMinError
        minEst[trial, EPS_FREQ] = minMinError
        maxEst[trial, EPS_FREQ] = maxMinError

        meanLdaOpt[trial, EPS_FREQ] = meanLdaIndex * ldaStep
        minLdaOpt[trial, EPS_FREQ] = minLdaIndex * ldaStep
        maxLdaOpt[trial, EPS_FREQ] = maxLdaIndex * ldaStep

        def error_bars(mLdaOpt, mUpper, mLower, tr, ef):
            """Method to set boolean values to determine error bars."""

            # no upper error bars when lda equals 0
            if mLdaOpt[tr, ef] == 0:
                mUpper[tr, ef] = 0
                mLower[tr, ef] = 1

            # no lower error bars when lda equals 1
            elif mLdaOpt[tr, ef] == 1:
                mLower[tr, ef] = 0
                mUpper[tr, ef] = 1

            # both error bars in all other cases
            else:
                mLower[tr, ef] = 1
                mUpper[tr, ef] = 1

        error_bars(meanLdaOpt, meanUpper, meanLower, trial, EPS_FREQ)
        error_bars(minLdaOpt, minUpper, minLower, trial, EPS_FREQ)
        error_bars(maxLdaOpt, maxUpper, maxLower, trial, EPS_FREQ)

        # option 2c: server adds Laplace noise term to final result
        if trial == 4 or trial == 5:
            lapNoise = tfp.distributions.Laplace(loc = A, scale = b1)

            # define error = squared difference between estimator and ground truth
            meanEst[trial, EPS_FREQ] = (meanEst[trial, EPS_FREQ] + lapNoise.sample(sample_shape = (1,)) - np.mean(uList))**2
            minEst[trial, EPS_FREQ] = (minEst[trial, EPS_FREQ] + lapNoise.sample(sample_shape = (1,)) - uList[minIndex])**2
            maxEst[trial, EPS_FREQ] = (maxEst[trial, EPS_FREQ] + lapNoise.sample(sample_shape = (1,)) - uList[maxIndex])**2
        
        # option 2a/b: clients or intermediate server already added Gaussian noise term
        else:
            meanEst[trial, EPS_FREQ] = (meanEst[trial, EPS_FREQ] - np.mean(uList))**2
            minEst[trial, EPS_FREQ] = (minEst[trial, EPS_FREQ] - uList[minIndex])**2
            maxEst[trial, EPS_FREQ] = (maxEst[trial, EPS_FREQ] - uList[maxIndex])**2

        statsfile.write(f"FEMNIST: Eps = {eps}\n")
        statsfile.write(f"Optimal Lambda {round(meanLdaOpt[trial, EPS_FREQ], 2)} for Mean Error {round(meanEst[trial, EPS_FREQ], 2)}\n")
        statsfile.write(f"Optimal Lambda {round(minLdaOpt[trial, EPS_FREQ], 2)} for Error {round(minEst[trial, EPS_FREQ], 2)} for Min Pair {minPair}\n")
        statsfile.write(f"Optimal Lambda {round(maxLdaOpt[trial, EPS_FREQ], 2)} for Error {round(maxEst[trial, EPS_FREQ], 2)} for Max Pair {maxPair}\n\n")

        EPS_FREQ = EPS_FREQ + 1

def change_marker(caps):
    """Method to change default arrow marker to bar."""
    for cap in caps:
        cap.set_marker('_')
        cap.set_markersize(10)

# plot error of unbiased estimator for each epsilon (mean)
fig, ax1 = plt.subplots()
plots1a, caps1a, bars1a = ax1.errorbar(epsset, meanEst[0], yerr = np.minimum(np.sqrt(meanEst[0]), np.divide(meanEst[0], 2)), color = 'blue', marker = 'o', label = 'start gauss')
plots1b, caps1b, bars1b = ax1.errorbar(epsset, meanEst[1], yerr = np.minimum(np.sqrt(meanEst[1]), np.divide(meanEst[1], 2)), color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plots1c, caps1c, bars1c = ax1.errorbar(epsset, meanEst[2], yerr = np.minimum(np.sqrt(meanEst[2]), np.divide(meanEst[2], 2)), color = 'green', marker = 'o', label = 'mid gauss')
plots1d, caps1d, bars1d = ax1.errorbar(epsset, meanEst[3], yerr = np.minimum(np.sqrt(meanEst[3]), np.divide(meanEst[3], 2)), color = 'lime', marker = 'x', label = 'mid gauss mc')
plots1e, caps1e, bars1e = ax1.errorbar(epsset, meanEst[4], yerr = np.minimum(np.sqrt(meanEst[4]), np.divide(meanEst[4], 2)), color = 'orange', marker = 'o', label = 'end lap')
plots1f, caps1f, bars1f = ax1.errorbar(epsset, meanEst[5], yerr = np.minimum(np.sqrt(meanEst[5]), np.divide(meanEst[5], 2)), color = 'gold', marker = 'x', label = 'end lap mc')
plots1g, caps1g, bars1g = ax1.errorbar(epsset, meanEst[6], yerr = np.minimum(np.sqrt(meanEst[6]), np.divide(meanEst[6], 2)), color = 'red', marker = '*', label = 'no noise')

change_marker(caps1a)
change_marker(caps1b)
change_marker(caps1c)
change_marker(caps1d)
change_marker(caps1e)
change_marker(caps1f)
change_marker(caps1g)

ax1.legend(loc = 'best')
ax1.set_yscale('log')
ax1.set_xlabel("Value of epsilon")
ax1.set_ylabel("Error of unbiased estimator (mean)")
ax1.set_title("How epsilon affects error of unbiased estimator (mean)")
fig.savefig("Femnist_pixel_eps_mean.png")
fig.clf()

# plot optimum lambda for each epsilon (mean)
print(f"\nmeanLdaOpt[3]: {meanLdaOpt[3]}")
print(f"\nmeanUpper[3]: {meanUpper[3]}")
print(f"\nmeanLower[3]: {meanLower[3]}")
fig, ax2 = plt.subplots()
plots2a, caps2a, bars2a = ax2.errorbar(epsset, meanLdaOpt[0], yerr = ldaStep, uplims = meanUpper[0], lolims = meanLower[0], color = 'blue', marker = 'o', label = 'start gauss')
plots2b, caps2b, bars2b = ax2.errorbar(epsset, meanLdaOpt[1], yerr = ldaStep, uplims = meanUpper[1], lolims = meanLower[1], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plots2c, caps2c, bars2c = ax2.errorbar(epsset, meanLdaOpt[2], yerr = ldaStep, uplims = meanUpper[2], lolims = meanLower[2], color = 'green', marker = 'o', label = 'mid gauss')
plots2d, caps2d, bars2d = ax2.errorbar(epsset, meanLdaOpt[3], yerr = ldaStep, uplims = meanUpper[3], lolims = meanLower[3], color = 'lime', marker = 'x', label = 'mid gauss mc')
plots2e, caps2e, bars2e = ax2.errorbar(epsset, meanLdaOpt[4], yerr = ldaStep, uplims = meanUpper[4], lolims = meanLower[4], color = 'orange', marker = 'o', label = 'end lap')
plots2f, caps2f, bars2f = ax2.errorbar(epsset, meanLdaOpt[5], yerr = ldaStep, uplims = meanUpper[5], lolims = meanLower[5], color = 'gold', marker = 'x', label = 'end lap mc')
plots2g, caps2g, bars2g = ax2.errorbar(epsset, meanLdaOpt[6], yerr = ldaStep, uplims = meanUpper[6], lolims = meanLower[6], color = 'red', marker = '*', label = 'no noise')

change_marker(caps2a)
change_marker(caps2b)
change_marker(caps2c)
change_marker(caps2d)
change_marker(caps2e)
change_marker(caps2f)
change_marker(caps2g)

ax2.legend(loc = 'best')
ax2.set_xlabel("Value of epsilon")
ax2.set_ylabel("Lambda to minimise error of unbiased estimator (mean)")
ax2.set_title("How epsilon affects optimum lambda (mean)")
fig.savefig("Femnist_pixel_eps_mean_lda.png")
fig.clf()

# plot error of unbiased estimator for each epsilon (min pair)
fig, ax3 = plt.subplots()
plots3a, caps3a, bars3a = ax3.errorbar(epsset, minEst[0], yerr = np.minimum(np.sqrt(minEst[0]), np.divide(minEst[0], 2)), color = 'blue', marker = 'o', label = 'start gauss')
plots3b, caps3b, bars3b = ax3.errorbar(epsset, minEst[1], yerr = np.minimum(np.sqrt(minEst[1]), np.divide(minEst[1], 2)), color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plots3c, caps3c, bars3c = ax3.errorbar(epsset, minEst[2], yerr = np.minimum(np.sqrt(minEst[2]), np.divide(minEst[2], 2)), color = 'green', marker = 'o', label = 'mid gauss')
plots3d, caps3d, bars3d = ax3.errorbar(epsset, minEst[3], yerr = np.minimum(np.sqrt(minEst[3]), np.divide(minEst[3], 2)), color = 'lime', marker = 'x', label = 'mid gauss mc')
plots3e, caps3e, bars3e = ax3.errorbar(epsset, minEst[4], yerr = np.minimum(np.sqrt(minEst[4]), np.divide(minEst[4], 2)), color = 'orange', marker = 'o', label = 'end lap')
plots3f, caps3f, bars3f = ax3.errorbar(epsset, minEst[5], yerr = np.minimum(np.sqrt(minEst[5]), np.divide(minEst[5], 2)), color = 'gold', marker = 'x', label = 'end lap mc')
plots3g, caps3g, bars3g = ax3.errorbar(epsset, minEst[6], yerr = np.minimum(np.sqrt(minEst[6]), np.divide(minEst[6], 2)), color = 'red', marker = '*', label = 'no noise')

change_marker(caps3a)
change_marker(caps3b)
change_marker(caps3c)
change_marker(caps3d)
change_marker(caps3e)
change_marker(caps3f)
change_marker(caps3g)

ax3.legend(loc = 'best')
ax3.set_yscale('log')
ax3.set_xlabel("Value of epsilon")
ax3.set_ylabel("Error of unbiased estimator (min pair)")
ax3.set_title("How epsilon affects error of unbiased estimator (min pair)")
fig.savefig("Femnist_pixel_eps_min.png")
fig.clf()

# plot optimum lambda for each epsilon (min pair)
print(f"\nminLdaOpt[3]: {minLdaOpt[3]}")
print(f"\nminUpper[3]: {minUpper[3]}")
print(f"\nminLower[3]: {minLower[3]}")
fig, ax4 = plt.subplots()
plots4a, caps4a, bars4a = ax4.errorbar(epsset, minLdaOpt[0], yerr = ldaStep, uplims = minUpper[0], lolims = minLower[0], color = 'blue', marker = 'o', label = 'start gauss')
plots4b, caps4b, bars4b = ax4.errorbar(epsset, minLdaOpt[1], yerr = ldaStep, uplims = minUpper[1], lolims = minLower[1], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plots4c, caps4c, bars4c = ax4.errorbar(epsset, minLdaOpt[2], yerr = ldaStep, uplims = minUpper[2], lolims = minLower[2], color = 'green', marker = 'o', label = 'mid gauss')
plots4d, caps4d, bars4d = ax4.errorbar(epsset, minLdaOpt[3], yerr = ldaStep, uplims = minUpper[3], lolims = minLower[3], color = 'lime', marker = 'x', label = 'mid gauss mc')
plots4e, caps4e, bars4e = ax4.errorbar(epsset, minLdaOpt[4], yerr = ldaStep, uplims = minUpper[4], lolims = minLower[4], color = 'orange', marker = 'o', label = 'end lap')
plots4f, caps4f, bars4f = ax4.errorbar(epsset, minLdaOpt[5], yerr = ldaStep, uplims = minUpper[5], lolims = minLower[5], color = 'gold', marker = 'x', label = 'end lap mc')
plots4g, caps4g, bars4g = ax4.errorbar(epsset, minLdaOpt[6], yerr = ldaStep, uplims = minUpper[6], lolims = minLower[6], color = 'red', marker = '*', label = 'no noise')

change_marker(caps4a)
change_marker(caps4b)
change_marker(caps4c)
change_marker(caps4d)
change_marker(caps4e)
change_marker(caps4f)
change_marker(caps4g)

ax4.legend(loc = 'best')
ax4.set_xlabel("Value of epsilon")
ax4.set_ylabel("Lambda to minimise error of unbiased estimator (min pair)")
ax4.set_title("How epsilon affects optimum lambda (min pair)")
fig.savefig("Femnist_pixel_eps_min_lda.png")
fig.clf()

# plot error of unbiased estimator for each epsilon (max pair)
fig, ax5 = plt.subplots()
plots5a, caps5a, bars5a = ax5.errorbar(epsset, maxEst[0], yerr = np.minimum(np.sqrt(maxEst[0]), np.divide(maxEst[0], 2)), color = 'blue', marker = 'o', label = 'start gauss')
plots5b, caps5b, bars5b = ax5.errorbar(epsset, maxEst[1], yerr = np.minimum(np.sqrt(maxEst[1]), np.divide(maxEst[1], 2)), color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plots5c, caps5c, bars5c = ax5.errorbar(epsset, maxEst[2], yerr = np.minimum(np.sqrt(maxEst[2]), np.divide(maxEst[2], 2)), color = 'green', marker = 'o', label = 'mid gauss')
plots5d, caps5d, bars5d = ax5.errorbar(epsset, maxEst[3], yerr = np.minimum(np.sqrt(maxEst[3]), np.divide(maxEst[3], 2)), color = 'lime', marker = 'x', label = 'mid gauss mc')
plots5e, caps5e, bars5e = ax5.errorbar(epsset, maxEst[4], yerr = np.minimum(np.sqrt(maxEst[4]), np.divide(maxEst[4], 2)), color = 'orange', marker = 'o', label = 'end lap')
plots5f, caps5f, bars5f = ax5.errorbar(epsset, maxEst[5], yerr = np.minimum(np.sqrt(maxEst[5]), np.divide(maxEst[5], 2)), color = 'gold', marker = 'x', label = 'end lap mc')
plots5g, caps5g, bars5g = ax5.errorbar(epsset, maxEst[6], yerr = np.minimum(np.sqrt(maxEst[6]), np.divide(maxEst[6], 2)), color = 'red', marker = '*', label = 'no noise')

change_marker(caps5a)
change_marker(caps5b)
change_marker(caps5c)
change_marker(caps5d)
change_marker(caps5e)
change_marker(caps5f)
change_marker(caps5g)

ax5.set_yscale('log')
ax5.set_xlabel("Value of epsilon")
ax5.set_ylabel("Error of unbiased estimator (max pair)")
ax5.set_title("How epsilon affects error of unbiased estimator (max pair)")
fig.savefig("Femnist_pixel_eps_max.png")
fig.clf()

# plot optimum lambda for each epsilon (max pair)
print(f"\nmaxLdaOpt[3]: {maxLdaOpt[3]}")
print(f"\nmaxUpper[3]: {maxUpper[3]}")
print(f"\nmaxLower[3]: {maxLower[3]}")
fig, ax6 = plt.subplots()
plots6a, caps6a, bars6a = ax6.errorbar(epsset, maxLdaOpt[0], yerr = ldaStep, uplims = maxUpper[0], lolims = maxLower[0], color = 'blue', marker = 'o', label = 'start gauss')
plots6b, caps6b, bars6b = ax6.errorbar(epsset, maxLdaOpt[1], yerr = ldaStep, uplims = maxUpper[1], lolims = maxLower[1], color = 'blueviolet', marker = 'x', label = 'start gauss mc')
plots6c, caps6c, bars6c = ax6.errorbar(epsset, maxLdaOpt[2], yerr = ldaStep, uplims = maxUpper[2], lolims = maxLower[2], color = 'green', marker = 'o', label = 'mid gauss')
plots6d, caps6d, bars6d = ax6.errorbar(epsset, maxLdaOpt[3], yerr = ldaStep, uplims = maxUpper[3], lolims = maxLower[3], color = 'lime', marker = 'x', label = 'mid gauss mc')
plots6e, caps6e, bars6e = ax6.errorbar(epsset, maxLdaOpt[4], yerr = ldaStep, uplims = maxUpper[4], lolims = maxLower[4], color = 'orange', marker = 'o', label = 'end lap')
plots6f, caps6f, bars6f = ax6.errorbar(epsset, maxLdaOpt[5], yerr = ldaStep, uplims = maxUpper[5], lolims = maxLower[5], color = 'gold', marker = 'x', label = 'end lap mc')
plots6g, caps6g, bars6g = ax6.errorbar(epsset, maxLdaOpt[6], yerr = ldaStep, uplims = maxUpper[6], lolims = maxLower[6], color = 'red', marker = '*', label = 'no noise')

change_marker(caps6a)
change_marker(caps6b)
change_marker(caps6c)
change_marker(caps6d)
change_marker(caps6e)
change_marker(caps6f)
change_marker(caps6g)

ax6.legend(loc = 'best')
ax6.set_xlabel("Value of epsilon")
ax6.set_ylabel("Lambda to minimise error of unbiased estimator (max pair)")
ax6.set_title("How epsilon affects optimum lambda (max pair)")
fig.savefig("Femnist_pixel_eps_max_lda.png")
fig.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
