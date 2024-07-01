"""Modules provide various time-related functions, compute the natural logarithm of a number,
create static, animated, and interactive visualisations, provide both a high- and low-level interface
to the HDF5 library, work with arrays, and carry out fast numerical computations in Python."""
import time
from math import log
import matplotlib as mpl
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

# lists of the values of epsilon and lambda, as well as the trials that will be explored
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist", "TAgg", "Trusted", "NoAlgo"]
ES = len(epsset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanValue = np.zeros((TS, ES))
meanEst = np.zeros((TS, ES))
meanLdaOpt = np.zeros((TS, ES))
meanEstZero = np.zeros((TS, ES))
meanEstOne = np.zeros((TS, ES))
meanPerc = np.zeros((TS, ES))
meanEpsSmall = np.zeros((TS, LS))
meanEpsDef = np.zeros((TS, LS))
meanEpsMid = np.zeros((TS, LS))
meanEpsLarge = np.zeros((TS, LS))

# related to min pairs
minValue = np.zeros((TS, ES))
minEst = np.zeros((TS, ES))
minLdaOpt = np.zeros((TS, ES))
minEstZero = np.zeros((TS, ES))
minEstOne = np.zeros((TS, ES))
minPerc = np.zeros((TS, ES))
minEpsSmall = np.zeros((TS, LS))
minEpsDef = np.zeros((TS, LS))
minEpsMid = np.zeros((TS, LS))
minEpsLarge = np.zeros((TS, LS))

# related to max pairs
maxValue = np.zeros((TS, ES))
maxEst = np.zeros((TS, ES))
maxLdaOpt = np.zeros((TS, ES))
maxEstZero = np.zeros((TS, ES))
maxEstOne = np.zeros((TS, ES))
maxPerc = np.zeros((TS, ES))
maxEpsSmall = np.zeros((TS, LS))
maxEpsDef = np.zeros((TS, LS))
maxEpsMid = np.zeros((TS, LS))
maxEpsLarge = np.zeros((TS, LS))

for trial in range(4):

    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    meanfile = open(f"femnist_eps_{trialset[trial]}_mean.txt", "w", encoding = 'utf-8')
    minfile = open(f"femnist_eps_{trialset[trial]}_min.txt", "w", encoding = 'utf-8')
    maxfile = open(f"femnist_eps_{trialset[trial]}_max.txt", "w", encoding = 'utf-8')
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

        # parameters for the addition of Laplace and Gaussian noise
        DTA = 0.1
        A = 0
        R1 = 90

        b1 = (1 + log(2)) / eps
        b2 = (2*((log(1.25))/DTA)*b1) / eps

        # load Gaussian noise distributions for clients and intermediate server
        if trial < 2:
            s = b2 * (np.sqrt(2) / R1)
            probGaussNoise = tfp.distributions.Normal(loc = A, scale = s / 100)
            gaussNoise = tfp.distributions.Normal(loc = A, scale = s)

        # stores for exact unknown distributions
        uDist = np.zeros((10, 10, U))
        nDist = np.zeros((10, 10, E))
        uList = []
        uCDList = []
        rList = []
        startNoise = []

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

                    # "Dist" (each client adds Gaussian noise term)
                    if trial == 0:
                        startSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                        startNoise.append(startSample)
                        ratio = ratio + startSample
                    
                    rList.append(ratio)
        
        # store for PRIEST-KLD
        R2 = len(rList)
        uEst = np.zeros((LS, R2))
        R_FREQ = 0

        for row in range(0, R2):
            uLogr = np.log(rList[row])
            LDA_FREQ = 0

            # explore lambdas in a range
            for lda in ldaset:

                # compute k3 estimator
                uRangeEst = lda * (np.exp(uLogr) - 1) - uLogr

                # share PRIEST-KLD with server
                uEst[LDA_FREQ, R_FREQ] = uRangeEst
                LDA_FREQ = LDA_FREQ + 1
            
            R_FREQ = R_FREQ + 1
        
        # extract position and identity of max and min pairs
        minIndex = np.argmin(uList)
        maxIndex = np.argmax(uList)
        minPair = uCDList[minIndex]
        maxPair = uCDList[maxIndex]

        # extract ground truths
        meanValue[trial, EPS_FREQ] = np.mean(uList)
        minValue[trial, EPS_FREQ] = uList[minIndex]
        maxValue[trial, EPS_FREQ] = uList[maxIndex]
        
        meanLda = np.zeros(LS)
        minLda = np.zeros(LS)
        maxLda = np.zeros(LS)

        meanLdaNoise = np.zeros(LS)
        minLdaNoise = np.zeros(LS)
        maxLdaNoise = np.zeros(LS)

        # compute mean error of PRIEST-KLD for each lambda
        for l in range(0, LS):
            meanLda[l] = np.mean(uEst[l])

            # extract error for max and min pairs
            minLda[l] = uEst[l, minIndex]
            maxLda[l] = uEst[l, maxIndex]

            def tagg(mldanoise, mlda, mepssmall, mepsdef, mepsmid, mepslarge):
                """TAgg: Intermediate server adds Gaussian noise term."""
                if trial == 1:
                    mldanoise[l] = gaussNoise.sample(sample_shape = (1,))
                    mlda[l] = mlda[l] + mldanoise[l]

                if EPS_FREQ == 0:
                    mepssmall[trial, l] = mlda[l]
                if EPS_FREQ == 3:
                    mepsdef[trial, l] = mlda[l]
                if EPS_FREQ == 6:
                    mepsmid[trial, l] = mlda[l]
                if EPS_FREQ == 10:
                    mepslarge[trial, l] = mlda[l]

            tagg(meanLdaNoise, meanLda, meanEpsSmall, meanEpsDef, meanEpsMid, meanEpsLarge)
            tagg(minLdaNoise, minLda, minEpsSmall, minEpsDef, minEpsMid, minEpsLarge)
            tagg(maxLdaNoise, maxLda, maxEpsSmall, maxEpsDef, maxEpsMid, maxEpsLarge)

        ldaStep = 0.05

        def opt_lda(mlda, mest, mldaopt, mestzero, mestone):
            """Method to find lambda that produces minimum error."""
            mldaindex = np.argmin(mlda)
            mminerror = mlda[mldaindex]
            mest[trial, EPS_FREQ] = mminerror
            mldaopt[trial, EPS_FREQ] = mldaindex * ldaStep

            # estimates for lambda = 0 and 1
            mestzero[trial, EPS_FREQ] = mlda[0]
            mestone[trial, EPS_FREQ] = mlda[LS-1]

        opt_lda(meanLda, meanEst, meanLdaOpt, meanEstZero, meanEstOne)
        opt_lda(minLda, minEst, minLdaOpt, minEstZero, minEstOne)
        opt_lda(maxLda, maxEst, maxLdaOpt, maxEstZero, maxEstOne)

        if trial == 2:
            lapNoise = tfp.distributions.Laplace(loc = A, scale = b1)
            meanNoise = lapNoise.sample(sample_shape = (1,))
            minNoise = lapNoise.sample(sample_shape = (1,))
            maxNoise = lapNoise.sample(sample_shape = (1,))

            def trusted(mest, mnoise, mestzero, mestone, mvalue):
                """Trusted: Server adds Laplace noise term to final result."""
                mzeronoise = lapNoise.sample(sample_shape = (1,))
                monenoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                mest[trial, EPS_FREQ] = (mest[trial, EPS_FREQ] + mnoise - mvalue[trial, EPS_FREQ])**2
                mestzero[trial, EPS_FREQ] = (mestzero[trial, EPS_FREQ] + mzeronoise - mvalue[trial, EPS_FREQ])**2
                mestone[trial, EPS_FREQ] = (mestone[trial, EPS_FREQ] + monenoise - mvalue[trial, EPS_FREQ])**2
            
            trusted(meanEst, meanNoise, meanValue, meanEstZero, meanEstOne)
            trusted(minEst, minNoise, minValue, minEstZero, minEstOne)
            trusted(maxEst, maxNoise, maxValue, maxEstZero, maxEstOne)

            for l in range(LS):
                
                def trusted_eps(mepssmall, mepsdef, mepsmid, mepslarge, mvalue):
                    """Trusted method for cases where eps is constant and lda is changing."""
                    # eps = 0.05 (small)
                    if EPS_FREQ == 0:
                        msmallnoise = lapNoise.sample(sample_shape = (1,))
                        mepssmall[trial, l] = (mepssmall[trial, l] + msmallnoise - mvalue[trial, EPS_FREQ])**2

                    # eps = 0.5 (def)
                    if EPS_FREQ == 3:
                        mdefnoise = lapNoise.sample(sample_shape = (1,))
                        mepsdef[trial, l] = (mepsdef[trial, l] + mdefnoise - mvalue[trial, EPS_FREQ])**2

                    # eps = 1.5 (mid)
                    if EPS_FREQ == 6:
                        mmidnoise = lapNoise.sample(sample_shape = (1,))
                        mepsmid[trial, l] = (mepsmid[trial, l] + mmidnoise - mvalue[trial, EPS_FREQ])**2

                    # eps = 3 (large)
                    if EPS_FREQ == 10:
                        mlargenoise = lapNoise.sample(sample_shape = (1,))
                        mepslarge[trial, l] = (mepslarge[trial, l] + mlargenoise - mvalue[trial, EPS_FREQ])**2

                trusted_eps(meanEpsSmall, meanEpsDef, meanEpsMid, meanEpsLarge, meanValue)
                trusted_eps(minEpsSmall, minEpsDef, minEpsMid, minEpsLarge, minValue)
                trusted_eps(maxEpsSmall, maxEpsDef, maxEpsMid, maxEpsLarge, maxValue)

        else:
            
            def trusted_else(mest, mvalue, mestzero, mestone, mepssmall, mepsdef, mepsmid, mepslarge):
                """Method for when clients or intermediate server already added Gaussian noise term."""
                mest[trial, EPS_FREQ] = (mest[trial, EPS_FREQ] - mvalue[trial, EPS_FREQ])**2
                mestzero[trial, EPS_FREQ] = (mestzero[trial, EPS_FREQ] - mvalue[trial, EPS_FREQ])**2
                mestone[trial, EPS_FREQ] = (mestone[trial, EPS_FREQ] - mvalue[trial, EPS_FREQ])**2

                for l in range(LS):
                    mepssmall[trial, l] = (mepssmall[trial, l] - mvalue[trial, EPS_FREQ])**2
                    mepsdef[trial, l] = (mepsdef[trial, l] - mvalue[trial, EPS_FREQ])**2
                    mepsmid[trial, l] = (mepsmid[trial, l] - mvalue[trial, EPS_FREQ])**2
                    mepslarge[trial, l] = (mepslarge[trial, l] - mvalue[trial, EPS_FREQ])**2
            
            trusted_else(meanEst, meanValue, meanEstZero, meanEstOne, meanEpsSmall, meanEpsDef, meanEpsMid, meanEpsLarge)
            trusted_else(minEst, minValue, minEstZero, minEstOne, minEpsSmall, minEpsDef, minEpsMid, minEpsLarge)
            trusted_else(maxEst, maxValue, maxEstZero, maxEstOne, maxEpsSmall, maxEpsDef, maxEpsMid, maxEpsLarge)

        if eps == epsset[0]:
            meanfile.write(f"FEMNIST: Eps = {eps}\n")
            minfile.write(f"FEMNIST: Eps = {eps}\n")
            maxfile.write(f"FEMNIST: Eps = {eps}\n")
        else:
            meanfile.write(f"\nEps = {eps}\n")
            minfile.write(f"\nEps = {eps}\n")
            maxfile.write(f"\nEps = {eps}\n")

        meanfile.write(f"\nMean Error: {round(meanEst[trial, EPS_FREQ], 2)}\n")
        meanfile.write(f"Optimal Lambda: {round(meanLdaOpt[trial, EPS_FREQ], 2)}\n")
        meanfile.write(f"Ground Truth: {round(meanValue[trial, EPS_FREQ], 2)}\n")

        # compute % of noise vs ground truth (mean)
        if trial == 0:
            meanPerc[trial, EPS_FREQ] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + meanValue[trial, EPS_FREQ])))*100
            meanfile.write(f"Noise: {np.round(meanPerc[trial, EPS_FREQ], 2)}%\n")
        if trial == 1:
            meanPerc[trial, EPS_FREQ] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + meanValue[trial, EPS_FREQ]))*100
            meanfile.write(f"Noise: {round(meanPerc[trial, EPS_FREQ], 2)}%\n")
        if trial == 2:
            meanPerc[trial, EPS_FREQ] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + meanValue[trial, EPS_FREQ])))*100
            meanfile.write(f"Noise: {np.round(meanPerc[trial, EPS_FREQ], 2)}%\n")

        minfile.write(f"\nMin Pair: {minPair}\n")
        minfile.write(f"Min Error: {round(minEst[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Optimal Lambda: {round(minLdaOpt[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Ground Truth: {round(minValue[trial, EPS_FREQ], 2)}\n")

        # compute % of noise vs ground truth (min pair)
        if trial == 0:
            minPerc[trial, EPS_FREQ] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + minValue[trial, EPS_FREQ])))*100
            minfile.write(f"Noise: {np.round(minPerc[trial, EPS_FREQ], 2)}%\n")
        if trial == 1:
            minPerc[trial, EPS_FREQ] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + minValue[trial, EPS_FREQ]))*100
            minfile.write(f"Noise: {round(minPerc[trial, EPS_FREQ], 2)}%\n")
        if trial == 2:
            minPerc[trial, EPS_FREQ] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + minValue[trial, EPS_FREQ])))*100
            minfile.write(f"Noise: {np.round(minPerc[trial, EPS_FREQ], 2)}%\n")

        maxfile.write(f"\nMax Pair: {maxPair}\n")
        maxfile.write(f"Max Error: {round(maxEst[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Optimal Lambda: {round(maxLdaOpt[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Ground Truth: {round(maxValue[trial, EPS_FREQ], 2)}\n")

        # compute % of noise vs ground truth (max pair) 
        if trial == 0:
            maxPerc[trial, EPS_FREQ] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + maxValue[trial, EPS_FREQ])))*100
            maxfile.write(f"Noise: {np.round(maxPerc[trial, EPS_FREQ], 2)}%\n")
        if trial == 1:
            maxPerc[trial, EPS_FREQ] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + maxValue[trial, EPS_FREQ]))*100
            maxfile.write(f"Noise: {round(maxPerc[trial, EPS_FREQ], 2)}%\n")
        if trial == 2:
            maxPerc[trial, EPS_FREQ] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + maxValue[trial, EPS_FREQ])))*100
            maxfile.write(f"Noise: {np.round(maxPerc[trial, EPS_FREQ], 2)}%\n")

        EPS_FREQ = EPS_FREQ + 1

# plot error of PRIEST-KLD for each epsilon (mean)
plt.errorbar(epsset, meanEst[0], yerr = np.minimum(np.sqrt(meanEst[0]), np.divide(meanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEst[1], yerr = np.minimum(np.sqrt(meanEst[1]), np.divide(meanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEst[2], yerr = np.minimum(np.sqrt(meanEst[2]), np.divide(meanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, meanEst[3], yerr = np.minimum(np.sqrt(meanEst[3]), np.divide(meanEst[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (min pair)
plt.errorbar(epsset, minEst[0], yerr = np.minimum(np.sqrt(minEst[0]), np.divide(minEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minEst[1], yerr = np.minimum(np.sqrt(minEst[1]), np.divide(minEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minEst[2], yerr = np.minimum(np.sqrt(minEst[2]), np.divide(minEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, minEst[3], yerr = np.minimum(np.sqrt(minEst[3]), np.divide(minEst[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (max pair)
plt.errorbar(epsset, maxEst[0], yerr = np.minimum(np.sqrt(maxEst[0]), np.divide(maxEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxEst[1], yerr = np.minimum(np.sqrt(maxEst[1]), np.divide(maxEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxEst[2], yerr = np.minimum(np.sqrt(maxEst[2]), np.divide(maxEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, maxEst[3], yerr = np.minimum(np.sqrt(maxEst[3]), np.divide(maxEst[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max.png")
plt.clf()

# plot optimum lambda for each epsilon (mean)
plt.plot(epsset, meanLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, meanLdaOpt[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, meanLdaOpt[2], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, meanLdaOpt[3], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_eps_lda_opt_mean.png")
plt.clf()

# plot optimum lambda for each epsilon (min pair)
plt.plot(epsset, minLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, minLdaOpt[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, minLdaOpt[2], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, minLdaOpt[3], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_eps_lda_opt_min.png")
plt.clf()

# plot optimum lambda for each epsilon (max pair)
plt.plot(epsset, maxLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, maxLdaOpt[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, maxLdaOpt[2], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(epsset, maxLdaOpt[3], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_eps_lda_opt_max.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 0 for each epsilon (mean)
plt.errorbar(epsset, meanEstZero[0], yerr = np.minimum(np.sqrt(meanEstZero[0]), np.divide(meanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstZero[1], yerr = np.minimum(np.sqrt(meanEstZero[1]), np.divide(meanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstZero[2], yerr = np.minimum(np.sqrt(meanEstZero[2]), np.divide(meanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, meanEstZero[3], yerr = np.minimum(np.sqrt(meanEstZero[3]), np.divide(meanEstZero[3], 2)),  color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (min pair)
plt.errorbar(epsset, minEstZero[0], yerr = np.minimum(np.sqrt(minEstZero[0]), np.divide(minEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minEstZero[1], yerr = np.minimum(np.sqrt(minEstZero[1]), np.divide(minEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minEstZero[2], yerr = np.minimum(np.sqrt(minEstZero[2]), np.divide(minEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, minEstZero[3], yerr = np.minimum(np.sqrt(minEstZero[3]), np.divide(minEstZero[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (max pair)
plt.errorbar(epsset, maxEstZero[0], yerr = np.minimum(np.sqrt(maxEstZero[0]), np.divide(maxEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxEstZero[1], yerr = np.minimum(np.sqrt(maxEstZero[1]), np.divide(maxEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxEstZero[2], yerr = np.minimum(np.sqrt(maxEstZero[2]), np.divide(maxEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, maxEstZero[3], yerr = np.minimum(np.sqrt(maxEstZero[3]), np.divide(maxEstZero[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_lda_zero.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 1 for each epsilon (mean)
plt.errorbar(epsset, meanEstOne[0], yerr = np.minimum(np.sqrt(meanEstOne[0]), np.divide(meanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstOne[1], yerr = np.minimum(np.sqrt(meanEstOne[1]), np.divide(meanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstOne[2], yerr = np.minimum(np.sqrt(meanEstOne[2]), np.divide(meanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, meanEstOne[3], yerr = np.minimum(np.sqrt(meanEstOne[3]), np.divide(meanEstOne[3], 2)),  color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (min pair)
plt.errorbar(epsset, minEstOne[0], yerr = np.minimum(np.sqrt(minEstOne[0]), np.divide(minEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minEstOne[1], yerr = np.minimum(np.sqrt(minEstOne[1]), np.divide(minEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minEstOne[2], yerr = np.minimum(np.sqrt(minEstOne[2]), np.divide(minEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, minEstOne[3], yerr = np.minimum(np.sqrt(minEstOne[3]), np.divide(minEstOne[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (max pair)
plt.errorbar(epsset, maxEstOne[0], yerr = np.minimum(np.sqrt(maxEstOne[0]), np.divide(maxEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxEstOne[1], yerr = np.minimum(np.sqrt(maxEstOne[1]), np.divide(maxEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxEstOne[2], yerr = np.minimum(np.sqrt(maxEstOne[2]), np.divide(maxEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(epsset, maxEstOne[3], yerr = np.minimum(np.sqrt(maxEstOne[3]), np.divide(maxEstOne[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (mean)
plt.errorbar(ldaset, meanEpsSmall[0], yerr = np.minimum(np.sqrt(meanEpsSmall[0]), np.divide(meanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsSmall[1], yerr = np.minimum(np.sqrt(meanEpsSmall[1]), np.divide(meanEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsSmall[2], yerr = np.minimum(np.sqrt(meanEpsSmall[2]), np.divide(meanEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsSmall[3], yerr = np.minimum(np.sqrt(meanEpsSmall[3]), np.divide(meanEpsSmall[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (min pair)
plt.errorbar(ldaset, minEpsSmall[0], yerr = np.minimum(np.sqrt(minEpsSmall[0]), np.divide(minEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsSmall[1], yerr = np.minimum(np.sqrt(minEpsSmall[1]), np.divide(minEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsSmall[2], yerr = np.minimum(np.sqrt(minEpsSmall[2]), np.divide(minEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsSmall[3], yerr = np.minimum(np.sqrt(minEpsSmall[3]), np.divide(minEpsSmall[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (max pair)
plt.errorbar(ldaset, maxEpsSmall[0], yerr = np.minimum(np.sqrt(maxEpsSmall[0]), np.divide(maxEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsSmall[1], yerr = np.minimum(np.sqrt(maxEpsSmall[1]), np.divide(maxEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsSmall[2], yerr = np.minimum(np.sqrt(maxEpsSmall[2]), np.divide(maxEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsSmall[3], yerr = np.minimum(np.sqrt(maxEpsSmall[3]), np.divide(maxEpsSmall[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (mean)
plt.errorbar(ldaset, meanEpsDef[0], yerr = np.minimum(np.sqrt(meanEpsDef[0]), np.divide(meanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsDef[1], yerr = np.minimum(np.sqrt(meanEpsDef[1]), np.divide(meanEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsDef[2], yerr = np.minimum(np.sqrt(meanEpsDef[2]), np.divide(meanEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsDef[3], yerr = np.minimum(np.sqrt(meanEpsDef[3]), np.divide(meanEpsDef[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (min pair)
plt.errorbar(ldaset, minEpsDef[0], yerr = np.minimum(np.sqrt(minEpsDef[0]), np.divide(minEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsDef[1], yerr = np.minimum(np.sqrt(minEpsDef[1]), np.divide(minEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsDef[2], yerr = np.minimum(np.sqrt(minEpsDef[2]), np.divide(minEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsDef[3], yerr = np.minimum(np.sqrt(minEpsDef[3]), np.divide(minEpsDef[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (max pair)
plt.errorbar(ldaset, maxEpsDef[0], yerr = np.minimum(np.sqrt(maxEpsDef[0]), np.divide(maxEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsDef[1], yerr = np.minimum(np.sqrt(maxEpsDef[1]), np.divide(maxEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsDef[2], yerr = np.minimum(np.sqrt(maxEpsDef[2]), np.divide(maxEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsDef[3], yerr = np.minimum(np.sqrt(maxEpsDef[3]), np.divide(maxEpsDef[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (mean)
plt.errorbar(ldaset, meanEpsMid[0], yerr = np.minimum(np.sqrt(meanEpsMid[0]), np.divide(meanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsMid[1], yerr = np.minimum(np.sqrt(meanEpsMid[1]), np.divide(meanEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsMid[2], yerr = np.minimum(np.sqrt(meanEpsMid[2]), np.divide(meanEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsMid[3], yerr = np.minimum(np.sqrt(meanEpsMid[3]), np.divide(meanEpsMid[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (min pair)
plt.errorbar(ldaset, minEpsMid[0], yerr = np.minimum(np.sqrt(minEpsMid[0]), np.divide(minEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsMid[1], yerr = np.minimum(np.sqrt(minEpsMid[1]), np.divide(minEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsMid[2], yerr = np.minimum(np.sqrt(minEpsMid[2]), np.divide(minEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsMid[3], yerr = np.minimum(np.sqrt(minEpsMid[3]), np.divide(minEpsMid[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (max pair)
plt.errorbar(ldaset, maxEpsMid[0], yerr = np.minimum(np.sqrt(maxEpsMid[0]), np.divide(maxEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsMid[1], yerr = np.minimum(np.sqrt(maxEpsMid[1]), np.divide(maxEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsMid[2], yerr = np.minimum(np.sqrt(maxEpsMid[2]), np.divide(maxEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsMid[3], yerr = np.minimum(np.sqrt(maxEpsMid[3]), np.divide(maxEpsMid[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (mean)
plt.errorbar(ldaset, meanEpsLarge[0], yerr = np.minimum(np.sqrt(meanEpsLarge[0]), np.divide(meanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsLarge[1], yerr = np.minimum(np.sqrt(meanEpsLarge[1]), np.divide(meanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsLarge[2], yerr = np.minimum(np.sqrt(meanEpsLarge[2]), np.divide(meanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsLarge[3], yerr = np.minimum(np.sqrt(meanEpsLarge[3]), np.divide(meanEpsLarge[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (min pair)
plt.errorbar(ldaset, minEpsLarge[0], yerr = np.minimum(np.sqrt(minEpsLarge[0]), np.divide(minEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsLarge[1], yerr = np.minimum(np.sqrt(minEpsLarge[1]), np.divide(minEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsLarge[2], yerr = np.minimum(np.sqrt(minEpsLarge[2]), np.divide(minEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsLarge[3], yerr = np.minimum(np.sqrt(minEpsLarge[3]), np.divide(minEpsLarge[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (max pair)
plt.errorbar(ldaset, maxEpsLarge[0], yerr = np.minimum(np.sqrt(maxEpsLarge[0]), np.divide(maxEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsLarge[1], yerr = np.minimum(np.sqrt(maxEpsLarge[1]), np.divide(maxEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsLarge[2], yerr = np.minimum(np.sqrt(maxEpsLarge[2]), np.divide(maxEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsLarge[3], yerr = np.minimum(np.sqrt(maxEpsLarge[3]), np.divide(maxEpsLarge[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_large.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (mean)
plt.plot(epsset, meanPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, meanPerc[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, meanPerc[2], color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([1, 10, 100, 1000, 10000])
plt.ylim(0.2, 10000)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_eps_perc_mean.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (min pair)
plt.plot(epsset, minPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, minPerc[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, minPerc[2], color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([1, 10, 100, 700])
plt.ylim(1, 700)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_eps_perc_min.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (max pair)
plt.plot(epsset, maxPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(epsset, maxPerc[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(epsset, maxPerc[2], color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([1, 10, 100, 500])
plt.ylim(0.1, 500)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_eps_perc_max.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
