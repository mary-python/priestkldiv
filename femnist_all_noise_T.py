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

# investigate samples from approx 1% to approx 20% of writers
Tset = [36, 72, 108, 144, 180, 225, 270, 360, 450, 540, 600, 660]
ES = len(Tset)

# list of the lambdas and trials that will be explored
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist", "TAgg", "Trusted", "NoAlgo"]
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanValue = np.zeros((TS, ES))
meanEst = np.zeros((TS, ES))
meanLdaOpt = np.zeros((TS, ES))
meanEstZero = np.zeros((TS, ES))
meanEstOne = np.zeros((TS, ES))
meanPerc = np.zeros((TS, ES))
meanTSmall = np.zeros((TS, LS))
meanTDef = np.zeros((TS, LS))
meanTMid = np.zeros((TS, LS))
meanTLarge = np.zeros((TS, LS))

# related to min pairs
minValue = np.zeros((TS, ES))
minEst = np.zeros((TS, ES))
minLdaOpt = np.zeros((TS, ES))
minEstZero = np.zeros((TS, ES))
minEstOne = np.zeros((TS, ES))
minPerc = np.zeros((TS, ES))
minTSmall = np.zeros((TS, LS))
minTDef = np.zeros((TS, LS))
minTMid = np.zeros((TS, LS))
minTLarge = np.zeros((TS, LS))

# related to max pairs
maxValue = np.zeros((TS, ES))
maxEst = np.zeros((TS, ES))
maxLdaOpt = np.zeros((TS, ES))
maxEstZero = np.zeros((TS, ES))
maxEstOne = np.zeros((TS, ES))
maxPerc = np.zeros((TS, ES))
maxTSmall = np.zeros((TS, LS))
maxTDef = np.zeros((TS, LS))
maxTMid = np.zeros((TS, LS))
maxTLarge = np.zeros((TS, LS))

for trial in range(4):
    
    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    meanfile = open(f"femnist_T_{trialset[trial]}_mean.txt", "w", encoding = 'utf-8')
    minfile = open(f"femnist_T_{trialset[trial]}_min.txt", "w", encoding = 'utf-8')
    maxfile = open(f"femnist_T_{trialset[trial]}_max.txt", "w", encoding = 'utf-8')
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
        
        # parameters for the addition of Laplace and Gaussian noise
        EPS = 0.5
        DTA = 0.1
        A = 0
        R1 = 90

        b1 = (1 + log(2)) / EPS
        b2 = (2*((log(1.25))/DTA)*b1) / EPS

        # load Gaussian noise distribution for intermediate server
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

                # share PRIEST-KLD with intermediate server
                uEst[LDA_FREQ, R_FREQ] = uRangeEst
                LDA_FREQ = LDA_FREQ + 1

            R_FREQ = R_FREQ + 1
        
        # extract position and identity of max and min pairs
        minIndex = np.argmin(uList)
        maxIndex = np.argmax(uList)
        minPair = uCDList[minIndex]
        maxPair = uCDList[maxIndex]

        # extract ground truths
        meanValue[trial, T_FREQ] = np.mean(uList)
        minValue[trial, T_FREQ] = uList[minIndex]
        maxValue[trial, T_FREQ] = uList[maxIndex]

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

            def tagg(mldanoise, mlda, mtsmall, mtdef, mtmid, mtlarge):
                """TAgg: Intermediate server adds Gaussian noise term."""
                if trial == 1:
                    mldanoise[l] = gaussNoise.sample(sample_shape = (1,))
                    mlda[l] = mlda[l] + mldanoise[l]

                if T_FREQ == 0:
                    mtsmall[trial, l] = mlda[l]
                if T_FREQ == 3:
                    mtdef[trial, l] = mlda[l]
                if T_FREQ == 6:
                    mtmid[trial, l] = mlda[l]
                if T_FREQ == 10:
                    mtlarge[trial, l] = mlda[l]

            tagg(meanLdaNoise, meanLda, meanTSmall, meanTDef, meanTMid, meanTLarge)
            tagg(minLdaNoise, minLda, minTSmall, minTDef, minTMid, minTLarge)
            tagg(maxLdaNoise, maxLda, maxTSmall, maxTDef, maxTMid, maxTLarge)

        ldaStep = 0.05

        def opt_lda(mlda, mest, mldaopt, mestzero, mestone):
            """Method to find lambda that produces minimum error."""
            mldaindex = np.argmin(mlda)
            mminerror = mlda[mldaindex]
            mest[trial, T_FREQ] = mminerror
            mldaopt[trial, T_FREQ] = mldaindex * ldaStep

            # estimates for lambda = 0 and 1
            mestzero[trial, T_FREQ] = mlda[0]
            mestone[trial, T_FREQ] = mlda[LS-1]

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
                mest[trial, T_FREQ] = (mest[trial, T_FREQ] + mnoise - mvalue[trial, T_FREQ])**2
                mestzero[trial, T_FREQ] = (mestzero[trial, T_FREQ] + mzeronoise - mvalue[trial, T_FREQ])**2
                mestone[trial, T_FREQ] = (mestone[trial, T_FREQ] + monenoise - mvalue[trial, T_FREQ])**2
            
            trusted(meanEst, meanNoise, meanValue, meanEstZero, meanEstOne)
            trusted(minEst, minNoise, minValue, minEstZero, minEstOne)
            trusted(maxEst, maxNoise, maxValue, maxEstZero, maxEstOne)

            for l in range(LS):
                
                def trusted_eps(mtsmall, mtdef, mtmid, mtlarge, mvalue):
                    """Trusted method for cases where eps is constant and lda is changing."""
                    # T = 36 (~1% of clients, small)
                    if T_FREQ == 0:
                        msmallnoise = lapNoise.sample(sample_shape = (1,))
                        mtsmall[trial, l] = (mtsmall[trial, l] + msmallnoise - mvalue[trial, T_FREQ])**2

                    # T = 180 (~5% of clients, default)
                    if T_FREQ == 3:
                        mdefnoise = lapNoise.sample(sample_shape = (1,))
                        mtdef[trial, l] = (mtdef[trial, l] + mdefnoise - mvalue[trial, T_FREQ])**2

                    # T = 360 (~10% of clients, mid)
                    if T_FREQ == 6:
                        mmidnoise = lapNoise.sample(sample_shape = (1,))
                        mtmid[trial, l] = (mtmid[trial, l] + mmidnoise - mvalue[trial, T_FREQ])**2

                    # T = 600 (~16.6% of clients, large)
                    if T_FREQ == 10:
                        mlargenoise = lapNoise.sample(sample_shape = (1,))
                        mtlarge[trial, l] = (mtlarge[trial, l] + mlargenoise - mvalue[trial, T_FREQ])**2

                trusted_eps(meanTSmall, meanTDef, meanTMid, meanTLarge, meanValue)
                trusted_eps(minTSmall, minTDef, minTMid, minTLarge, minValue)
                trusted_eps(maxTSmall, maxTDef, maxTMid, maxTLarge, maxValue)

        else:
            
            def trusted_else(mest, mvalue, mestzero, mestone, mtsmall, mtdef, mtmid, mtlarge):
                """Method for when clients or intermediate server already added Gaussian noise term."""
                mest[trial, T_FREQ] = (mest[trial, T_FREQ] - mvalue[trial, T_FREQ])**2
                mestzero[trial, T_FREQ] = (mestzero[trial, T_FREQ] - mvalue[trial, T_FREQ])**2
                mestone[trial, T_FREQ] = (mestone[trial, T_FREQ] - mvalue[trial, T_FREQ])**2

                for l in range(LS):
                    mtsmall[trial, l] = (mtsmall[trial, l] - mvalue[trial, T_FREQ])**2
                    mtdef[trial, l] = (mtdef[trial, l] - mvalue[trial, T_FREQ])**2
                    mtmid[trial, l] = (mtmid[trial, l] - mvalue[trial, T_FREQ])**2
                    mtlarge[trial, l] = (mtlarge[trial, l] - mvalue[trial, T_FREQ])**2
            
            trusted_else(meanEst, meanValue, meanEstZero, meanEstOne, meanTSmall, meanTDef, meanTMid, meanTLarge)
            trusted_else(minEst, minValue, minEstZero, minEstOne, minTSmall, minTDef, minTMid, minTLarge)
            trusted_else(maxEst, maxValue, maxEstZero, maxEstOne, maxTSmall, maxTDef, maxTMid, maxTLarge)

        if T == Tset[0]:
            meanfile.write(f"FEMNIST: T = {T}\n")
            minfile.write(f"FEMNIST: T = {T}\n")
            maxfile.write(f"FEMNIST: T = {T}\n")
        else:
            meanfile.write(f"\nT = {T}\n")
            minfile.write(f"\nT = {T}\n")
            maxfile.write(f"\nT = {T}\n")
        
        meanfile.write(f"\nMean Error: {round(meanEst[trial, T_FREQ], 2)}\n")
        meanfile.write(f"Optimal Lambda: {round(meanLdaOpt[trial, T_FREQ], 2)}\n")
        meanfile.write(f"Ground Truth: {round(meanValue[trial, T_FREQ], 2)}\n")

        # compute % of noise vs ground truth (mean)
        if trial == 0:
            meanPerc[trial, T_FREQ] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + meanValue[trial, T_FREQ])))*100
            meanfile.write(f"Noise: {np.round(meanPerc[trial, T_FREQ], 2)}%\n")
        if trial == 1:
            meanPerc[trial, T_FREQ] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + meanValue[trial, T_FREQ]))*100
            meanfile.write(f"Noise: {round(meanPerc[trial, T_FREQ], 2)}%\n")
        if trial == 2:
            meanPerc[trial, T_FREQ] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + meanValue[trial, T_FREQ])))*100
            meanfile.write(f"Noise: {np.round(meanPerc[trial, T_FREQ], 2)}%\n")

        minfile.write(f"\nMin Pair: {minPair}\n")
        minfile.write(f"Min Error: {round(minEst[trial, T_FREQ], 2)}\n")
        minfile.write(f"Optimal Lambda: {round(minLdaOpt[trial, T_FREQ], 2)}\n")
        minfile.write(f"Ground Truth: {round(minValue[trial, T_FREQ], 2)}\n")

        # compute % of noise vs ground truth (min pair)
        if trial == 0:
            minPerc[trial, T_FREQ] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + minValue[trial, T_FREQ])))*100
            minfile.write(f"Noise: {np.round(minPerc[trial, T_FREQ], 2)}%\n")
        if trial == 1:
            minPerc[trial, T_FREQ] = abs((np.sum(minLdaNoise)) / (np.sum(minLdaNoise) + minValue[trial, T_FREQ]))*100
            minfile.write(f"Noise: {round(minPerc[trial, T_FREQ], 2)}%\n")
        if trial == 2:
            minPerc[trial, T_FREQ] = float(abs(np.array(minNoise) / (np.array(minNoise) + minValue[trial, T_FREQ])))*100
            minfile.write(f"Noise: {np.round(minPerc[trial, T_FREQ], 2)}%\n")

        minfile.write(f"\nMax Pair: {maxPair}\n")
        maxfile.write(f"Max Error: {round(maxEst[trial, T_FREQ], 2)}\n")
        maxfile.write(f"Optimal Lambda: {round(maxLdaOpt[trial, T_FREQ], 2)}\n")
        maxfile.write(f"Ground Truth: {round(maxValue[trial, T_FREQ], 2)}\n")

        # compute % of noise vs ground truth (max pair)
        if trial == 0:
            maxPerc[trial, T_FREQ] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + maxValue[trial, T_FREQ])))*100
            maxfile.write(f"Noise: {np.round(maxPerc[trial, T_FREQ], 2)}%\n")
        if trial == 1:
            maxPerc[trial, T_FREQ] = abs((np.sum(maxLdaNoise)) / (np.sum(maxLdaNoise) + maxValue[trial, T_FREQ]))*100
            maxfile.write(f"Noise: {round(maxPerc[trial, T_FREQ], 2)}%\n")
        if trial == 2:
            maxPerc[trial, T_FREQ] = float(abs(np.array(maxNoise) / (np.array(maxNoise) + maxValue[trial, T_FREQ])))*100
            maxfile.write(f"Noise: {np.round(maxPerc[trial, T_FREQ], 2)}%\n")

        T_FREQ = T_FREQ + 1

# plot error of PRIEST-KLD for each T (mean)
plt.errorbar(Tset, meanEst[0], yerr = np.minimum(np.sqrt(meanEst[0]), np.divide(meanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanEst[1], yerr = np.minimum(np.sqrt(meanEst[1]), np.divide(meanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanEst[2], yerr = np.minimum(np.sqrt(meanEst[2]), np.divide(meanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanEst[3], yerr = np.minimum(np.sqrt(meanEst[3]), np.divide(meanEst[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean.png")
plt.clf()

# plot error of PRIEST-KLD for each T (min pair)
plt.errorbar(Tset, minEst[0], yerr = np.minimum(np.sqrt(minEst[0]), np.divide(minEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minEst[1], yerr = np.minimum(np.sqrt(minEst[1]), np.divide(minEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minEst[2], yerr = np.minimum(np.sqrt(minEst[2]), np.divide(minEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minEst[3], yerr = np.minimum(np.sqrt(minEst[3]), np.divide(minEst[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min.png")
plt.clf()

# plot error of PRIEST-KLD for each T (max pair)
plt.errorbar(Tset, maxEst[0], yerr = np.minimum(np.sqrt(maxEst[0]), np.divide(maxEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxEst[1], yerr = np.minimum(np.sqrt(maxEst[1]), np.divide(maxEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxEst[2], yerr = np.minimum(np.sqrt(maxEst[2]), np.divide(maxEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxEst[3], yerr = np.minimum(np.sqrt(maxEst[3]), np.divide(maxEst[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max.png")
plt.clf()

# plot optimum lambda for each T (mean)
plt.plot(Tset, meanLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Tset, meanLdaOpt[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Tset, meanLdaOpt[2], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Tset, meanLdaOpt[3], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_T_lda_opt_mean.png")
plt.clf()

# plot optimum lambda for each T (min pair)
plt.plot(Tset, minLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Tset, minLdaOpt[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Tset, minLdaOpt[2], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Tset, minLdaOpt[3], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_T_lda_opt_min.png")
plt.clf()

# plot optimum lambda for each T (max pair)
plt.plot(Tset, maxLdaOpt[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Tset, maxLdaOpt[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Tset, maxLdaOpt[2], color = 'orange', marker = 'o', label = "Trusted")
plt.plot(Tset, maxLdaOpt[3], color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_T_lda_opt_max.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 0 for each T (mean)
plt.errorbar(Tset, meanEstZero[0], yerr = np.minimum(np.sqrt(meanEstZero[0]), np.divide(meanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanEstZero[1], yerr = np.minimum(np.sqrt(meanEstZero[1]), np.divide(meanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanEstZero[2], yerr = np.minimum(np.sqrt(meanEstZero[2]), np.divide(meanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanEstZero[3], yerr = np.minimum(np.sqrt(meanEstZero[3]), np.divide(meanEstZero[3], 2)),  color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (min pair)
plt.errorbar(Tset, minEstZero[0], yerr = np.minimum(np.sqrt(minEstZero[0]), np.divide(minEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minEstZero[1], yerr = np.minimum(np.sqrt(minEstZero[1]), np.divide(minEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minEstZero[2], yerr = np.minimum(np.sqrt(minEstZero[2]), np.divide(minEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minEstZero[3], yerr = np.minimum(np.sqrt(minEstZero[3]), np.divide(minEstZero[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each T (max pair)
plt.errorbar(Tset, maxEstZero[0], yerr = np.minimum(np.sqrt(maxEstZero[0]), np.divide(maxEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxEstZero[1], yerr = np.minimum(np.sqrt(maxEstZero[1]), np.divide(maxEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxEstZero[2], yerr = np.minimum(np.sqrt(maxEstZero[2]), np.divide(maxEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxEstZero[3], yerr = np.minimum(np.sqrt(maxEstZero[3]), np.divide(maxEstZero[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max_lda_zero.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 1 for each T (mean)
plt.errorbar(Tset, meanEstOne[0], yerr = np.minimum(np.sqrt(meanEstOne[0]), np.divide(meanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanEstOne[1], yerr = np.minimum(np.sqrt(meanEstOne[1]), np.divide(meanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanEstOne[2], yerr = np.minimum(np.sqrt(meanEstOne[2]), np.divide(meanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanEstOne[3], yerr = np.minimum(np.sqrt(meanEstOne[3]), np.divide(meanEstOne[3], 2)),  color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (min pair)
plt.errorbar(Tset, minEstOne[0], yerr = np.minimum(np.sqrt(minEstOne[0]), np.divide(minEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minEstOne[1], yerr = np.minimum(np.sqrt(minEstOne[1]), np.divide(minEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minEstOne[2], yerr = np.minimum(np.sqrt(minEstOne[2]), np.divide(minEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minEstOne[3], yerr = np.minimum(np.sqrt(minEstOne[3]), np.divide(minEstOne[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each T (max pair)
plt.errorbar(Tset, maxEstOne[0], yerr = np.minimum(np.sqrt(maxEstOne[0]), np.divide(maxEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxEstOne[1], yerr = np.minimum(np.sqrt(maxEstOne[1]), np.divide(maxEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxEstOne[2], yerr = np.minimum(np.sqrt(maxEstOne[2]), np.divide(maxEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxEstOne[3], yerr = np.minimum(np.sqrt(maxEstOne[3]), np.divide(maxEstOne[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when T = 36 (mean)
plt.errorbar(ldaset, meanTSmall[0], yerr = np.minimum(np.sqrt(meanTSmall[0]), np.divide(meanTSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTSmall[1], yerr = np.minimum(np.sqrt(meanTSmall[1]), np.divide(meanTSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTSmall[2], yerr = np.minimum(np.sqrt(meanTSmall[2]), np.divide(meanTSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTSmall[3], yerr = np.minimum(np.sqrt(meanTSmall[3]), np.divide(meanTSmall[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean_T_small.png")
plt.clf()

# plot error of PRIEST-KLD when T = 36 (min pair)
plt.errorbar(ldaset, minTSmall[0], yerr = np.minimum(np.sqrt(minTSmall[0]), np.divide(minTSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTSmall[1], yerr = np.minimum(np.sqrt(minTSmall[1]), np.divide(minTSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTSmall[2], yerr = np.minimum(np.sqrt(minTSmall[2]), np.divide(minTSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTSmall[3], yerr = np.minimum(np.sqrt(minTSmall[3]), np.divide(minTSmall[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min_T_small.png")
plt.clf()

# plot error of PRIEST-KLD when T = 36 (max pair)
plt.errorbar(ldaset, maxTSmall[0], yerr = np.minimum(np.sqrt(maxTSmall[0]), np.divide(maxTSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTSmall[1], yerr = np.minimum(np.sqrt(maxTSmall[1]), np.divide(maxTSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTSmall[2], yerr = np.minimum(np.sqrt(maxTSmall[2]), np.divide(maxTSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTSmall[3], yerr = np.minimum(np.sqrt(maxTSmall[3]), np.divide(maxTSmall[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max_T_small.png")
plt.clf()

# plot error of PRIEST-KLD when T = 180 (mean)
plt.errorbar(ldaset, meanTDef[0], yerr = np.minimum(np.sqrt(meanTDef[0]), np.divide(meanTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTDef[1], yerr = np.minimum(np.sqrt(meanTDef[1]), np.divide(meanTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTDef[2], yerr = np.minimum(np.sqrt(meanTDef[2]), np.divide(meanTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTDef[3], yerr = np.minimum(np.sqrt(meanTDef[3]), np.divide(meanTDef[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean_T_def.png")
plt.clf()

# plot error of PRIEST-KLD when T = 180 (min pair)
plt.errorbar(ldaset, minTDef[0], yerr = np.minimum(np.sqrt(minTDef[0]), np.divide(minTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTDef[1], yerr = np.minimum(np.sqrt(minTDef[1]), np.divide(minTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTDef[2], yerr = np.minimum(np.sqrt(minTDef[2]), np.divide(minTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTDef[3], yerr = np.minimum(np.sqrt(minTDef[3]), np.divide(minTDef[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min_T_def.png")
plt.clf()

# plot error of PRIEST-KLD when T = 180 (max pair)
plt.errorbar(ldaset, maxTDef[0], yerr = np.minimum(np.sqrt(maxTDef[0]), np.divide(maxTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTDef[1], yerr = np.minimum(np.sqrt(maxTDef[1]), np.divide(maxTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTDef[2], yerr = np.minimum(np.sqrt(maxTDef[2]), np.divide(maxTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTDef[3], yerr = np.minimum(np.sqrt(maxTDef[3]), np.divide(maxTDef[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max_T_def.png")
plt.clf()

# plot error of PRIEST-KLD when T = 360 (mean)
plt.errorbar(ldaset, meanTMid[0], yerr = np.minimum(np.sqrt(meanTMid[0]), np.divide(meanTMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTMid[1], yerr = np.minimum(np.sqrt(meanTMid[1]), np.divide(meanTMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTMid[2], yerr = np.minimum(np.sqrt(meanTMid[2]), np.divide(meanTMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTMid[3], yerr = np.minimum(np.sqrt(meanTMid[3]), np.divide(meanTMid[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean_T_mid.png")
plt.clf()

# plot error of PRIEST-KLD when T = 360 (min pair)
plt.errorbar(ldaset, minTMid[0], yerr = np.minimum(np.sqrt(minTMid[0]), np.divide(minTMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTMid[1], yerr = np.minimum(np.sqrt(minTMid[1]), np.divide(minTMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTMid[2], yerr = np.minimum(np.sqrt(minTMid[2]), np.divide(minTMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTMid[3], yerr = np.minimum(np.sqrt(minTMid[3]), np.divide(minTMid[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min_T_mid.png")
plt.clf()

# plot error of PRIEST-KLD when T = 360 (max pair)
plt.errorbar(ldaset, maxTMid[0], yerr = np.minimum(np.sqrt(maxTMid[0]), np.divide(maxTMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTMid[1], yerr = np.minimum(np.sqrt(maxTMid[1]), np.divide(maxTMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTMid[2], yerr = np.minimum(np.sqrt(maxTMid[2]), np.divide(maxTMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTMid[3], yerr = np.minimum(np.sqrt(maxTMid[3]), np.divide(maxTMid[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max_T_mid.png")
plt.clf()

# plot error of PRIEST-KLD when T = 600 (mean)
plt.errorbar(ldaset, meanTLarge[0], yerr = np.minimum(np.sqrt(meanTLarge[0]), np.divide(meanTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTLarge[1], yerr = np.minimum(np.sqrt(meanTLarge[1]), np.divide(meanTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTLarge[2], yerr = np.minimum(np.sqrt(meanTLarge[2]), np.divide(meanTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTLarge[3], yerr = np.minimum(np.sqrt(meanTLarge[3]), np.divide(meanTLarge[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_mean_T_large.png")
plt.clf()

# plot error of PRIEST-KLD when T = 600 (min pair)
plt.errorbar(ldaset, minTLarge[0], yerr = np.minimum(np.sqrt(minTLarge[0]), np.divide(minTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTLarge[1], yerr = np.minimum(np.sqrt(minTLarge[1]), np.divide(minTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTLarge[2], yerr = np.minimum(np.sqrt(minTLarge[2]), np.divide(minTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTLarge[3], yerr = np.minimum(np.sqrt(minTLarge[3]), np.divide(minTLarge[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_min_T_large.png")
plt.clf()

# plot error of PRIEST-KLD when T = 600 (max pair)
plt.errorbar(ldaset, maxTLarge[0], yerr = np.minimum(np.sqrt(maxTLarge[0]), np.divide(maxTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTLarge[1], yerr = np.minimum(np.sqrt(maxTLarge[1]), np.divide(maxTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTLarge[2], yerr = np.minimum(np.sqrt(maxTLarge[2]), np.divide(maxTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTLarge[3], yerr = np.minimum(np.sqrt(maxTLarge[3]), np.divide(maxTLarge[3], 2)), color = 'red', marker = '*', label = "NoAlgo")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_T_est_max_T_large.png")
plt.clf()

# plot % of noise vs ground truth for each T (mean)
plt.plot(Tset, meanPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Tset, meanPerc[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Tset, meanPerc[2], color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([10, 100, 1000])
plt.ylim(8, 1000)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of T")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_T_perc_mean.png")
plt.clf()

# plot % of noise vs ground truth for each T (min pair)
plt.plot(Tset, minPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Tset, minPerc[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Tset, minPerc[2], color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([30, 100, 1000, 6000])
plt.ylim(30, 6000)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of T")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_T_perc_min.png")
plt.clf()

# plot % of noise vs ground truth for each T (max pair)
plt.plot(Tset, maxPerc[0], color = 'blue', marker = 'o', label = "Dist")
plt.plot(Tset, maxPerc[1], color = 'green', marker = 'o', label = "TAgg")
plt.plot(Tset, maxPerc[2], color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([2, 10, 100, 600])
plt.ylim(2, 600)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of T")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_T_perc_max.png")
plt.clf()

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"\nRuntime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"\nRuntime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
