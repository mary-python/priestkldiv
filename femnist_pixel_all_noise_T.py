"""Modules provide various time-related functions, compute the natural logarithm of a number, 
remember the order in which items are added, create static, animated, and interactive visualisations,
compute the mean of a list quickly and accurately, provide both a high- and low-level interface to
the HDF5 library, work with arrays, and carry out fast numerical computations in Python."""
import time
from math import log
from collections import OrderedDict
import matplotlib.pyplot as plt
from statistics import fmean
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

# list of the trials that will be run
trialset = ["end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc", "mid_gauss", "mid_gauss_mc"]
TS = len(trialset)

# investigate samples from approx 1% to approx 20% of writers
Tset = [36, 72, 108, 144, 180, 225, 270, 360, 450, 540, 600, 660]
ES = len(Tset)

# stores for mean of unbiased estimator and optimum lambda
meanEst = np.zeros((TS, ES))
meanLdaOpt = np.zeros((TS, ES))

# for min and max pairs
minEst = np.zeros((TS, ES))
minLdaOpt = np.zeros((TS, ES))
maxEst = np.zeros((TS, ES))
maxLdaOpt = np.zeros((TS, ES))

for trial in range(6):
    
    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    statsfile = open(f"femnist_{trialset[trial]}_noise.txt", "w", encoding = 'utf-8')
    T_FREQ = 0

    for T in Tset:
        print(f"Trial {trial + 1}: T = {T}...")

        # store T images corresponding to each digit
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

        # smoothing parameter: 0.1 and 1 are too large
        ALPHA = 0.01

        def smoothed_prob(dset, dig, im, ufreq):
            """Method to compute frequencies of unique images and return smoothed probabilities."""
            where = np.where(np.all(im == dset, axis = (1, 2)))
            freq = len(where[0])
            uImageSet[dig, ufreq] = im
            uFreqSet[dig, ufreq] = int(freq)
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

        # for each comparison digit compute exact unknown distributions for all digits
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

        # parameters for the addition of Laplace and Gaussian noise
        EPS = 0.1
        DTA = 0.1
        A = 0
        R = len(rList)

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

                # share unbiased estimator with intermediate server
                uEst[LDA_FREQ, R_FREQ] = uRangeEst
                LDA_FREQ = LDA_FREQ + 1

            R_FREQ = R_FREQ + 1

        # option 1a: baseline case
        if trial % 2 == 0:
            b1 = log(2) / EPS

        # option 1b: Monte Carlo estimate
        else:
            b1 = (1 + log(2)) / EPS

        b2 = (2*((log(1.25))/DTA)*b1) / EPS

        # load Gaussian noise distribution for intermediate server
        if trial >= 4:
            s = b2 * (np.sqrt(2) / R)
            midNoise = tfp.distributions.Normal(loc = A, scale = s)
        
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
            
            # option 2a: intermediate server adds noise term
            if trial >= 4:
                meanLda[l] = np.mean(uEst[l]) + midNoise.sample(sample_shape = (1,))

                # extract error for max and min pairs
                minLda[l] = uEst[l, minIndex] + midNoise.sample(sample_shape = (1,))
                maxLda[l] = uEst[l, maxIndex] + midNoise.sample(sample_shape = (1,))

            # option 2b: no noise until end
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
        meanEst[trial, T_FREQ] = meanMinError
        minEst[trial, T_FREQ] = minMinError
        maxEst[trial, T_FREQ] = maxMinError

        meanLdaOpt[trial, T_FREQ] = meanLdaIndex * ldaStep
        minLdaOpt[trial, T_FREQ] = minLdaIndex * ldaStep
        maxLdaOpt[trial, T_FREQ] = maxLdaIndex * ldaStep

        # option 2b: server adds noise term to final result
        if trial < 4:
            
            # option 3a: add Laplace noise
            if trial < 2:
                endNoise = tfp.distributions.Laplace(loc = A, scale = b1)
            
            # option 3b: add Gaussian noise
            else:
                endNoise = tfp.distributions.Normal(loc = A, scale = b2)

            meanEst[trial, T_FREQ] = meanEst[trial, T_FREQ] + endNoise.sample(sample_shape = (1,))
            minEst[trial, T_FREQ] = minEst[trial, T_FREQ] + endNoise.sample(sample_shape = (1,))
            maxEst[trial, T_FREQ] = maxEst[trial, T_FREQ] + endNoise.sample(sample_shape = (1,))

        statsfile.write(f"FEMNIST: T = {T}\n")
        statsfile.write(f"Optimal Lambda {round(meanLdaOpt[trial, T_FREQ], 4)} for Mean {round(meanEst[trial, T_FREQ], 4)}\n")
        statsfile.write(f"Optimal Lambda {round(minLdaOpt[trial, T_FREQ], 4)} for Error {round(minEst[trial, T_FREQ], 4)} for Min Pair {minPair}\n")
        statsfile.write(f"Optimal Lambda {round(maxLdaOpt[trial, T_FREQ], 4)} for Error {round(maxEst[trial, T_FREQ], 4)} for Max Pair {maxPair}\n\n")

        T_FREQ = T_FREQ + 1

# plot error of unbiased estimator for each T (mean)
print(f"\nmeanEst: {meanEst[0]}")
plt.errorbar(Tset, meanEst[0], yerr = np.std(meanEst[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, meanEst[1], yerr = np.std(meanEst[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, meanEst[2], yerr = np.std(meanEst[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, meanEst[3], yerr = np.std(meanEst[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, meanEst[4], yerr = np.std(meanEst[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, meanEst[5], yerr = np.std(meanEst[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (mean)")
plt.title("How T affects error of unbiased estimator (mean)")
plt.savefig("Femnist_pixel_T_mean.png")
plt.clf()

# plot optimum lambda for each T (mean)
print(f"\nmeanLdaOpt: {meanLdaOpt[0]}")
plt.errorbar(Tset, meanLdaOpt[0], yerr = np.std(meanLdaOpt[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, meanLdaOpt[1], yerr = np.std(meanLdaOpt[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, meanLdaOpt[2], yerr = np.std(meanLdaOpt[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, meanLdaOpt[3], yerr = np.std(meanLdaOpt[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, meanLdaOpt[4], yerr = np.std(meanLdaOpt[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, meanLdaOpt[5], yerr = np.std(meanLdaOpt[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xscale('log')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of unbiased estimator (mean)")
plt.title("How T affects optimum lambda (mean)")
plt.savefig("Femnist_pixel_T_mean_lda.png")
plt.clf()

# plot error of unbiased estimator for each T (min pair)
print(f"\nminEst: {minEst[0]}")
plt.errorbar(Tset, minEst[0], yerr = np.std(minEst[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, minEst[1], yerr = np.std(minEst[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, minEst[2], yerr = np.std(minEst[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, minEst[3], yerr = np.std(minEst[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, minEst[4], yerr = np.std(minEst[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, minEst[5], yerr = np.std(minEst[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (min pair)")
plt.title("How T affects error of unbiased estimator (min pair)")
plt.savefig("Femnist_pixel_T_min.png")
plt.clf()

# plot optimum lambda for each T (min pair)
print(f"\nminLdaOpt: {minLdaOpt[0]}")
plt.errorbar(Tset, minLdaOpt[0], yerr = np.std(minLdaOpt[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, minLdaOpt[1], yerr = np.std(minLdaOpt[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, minLdaOpt[2], yerr = np.std(minLdaOpt[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, minLdaOpt[3], yerr = np.std(minLdaOpt[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, minLdaOpt[4], yerr = np.std(minLdaOpt[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, minLdaOpt[5], yerr = np.std(minLdaOpt[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xscale('log')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of unbiased estimator (min pair)")
plt.title("How T affects optimum lambda (min pair)")
plt.savefig("Femnist_pixel_T_min_lda.png")
plt.clf()

# plot error of unbiased estimator for each T (max pair)
print(f"\nmaxEst: {maxEst[0]}")
plt.errorbar(Tset, maxEst[0], yerr = np.std(maxEst[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, maxEst[1], yerr = np.std(maxEst[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, maxEst[2], yerr = np.std(maxEst[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, maxEst[3], yerr = np.std(maxEst[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, maxEst[4], yerr = np.std(maxEst[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, maxEst[5], yerr = np.std(maxEst[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (max pair)")
plt.title("How T affects error of unbiased estimator (max pair)")
plt.savefig("Femnist_pixel_T_max.png")
plt.clf()

# plot optimum lambda for each T (mean)
print(f"\nmaxLdaOpt: {maxLdaOpt[0]}")
plt.errorbar(Tset, maxLdaOpt[0], yerr = np.std(maxLdaOpt[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, maxLdaOpt[1], yerr = np.std(maxLdaOpt[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, maxLdaOpt[2], yerr = np.std(maxLdaOpt[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, maxLdaOpt[3], yerr = np.std(maxLdaOpt[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, maxLdaOpt[4], yerr = np.std(maxLdaOpt[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, maxLdaOpt[5], yerr = np.std(maxLdaOpt[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xscale('log')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of unbiased estimator (max pair)")
plt.title("How T affects optimum lambda (max pair)")
plt.savefig("Femnist_pixel_T_max_lda.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
