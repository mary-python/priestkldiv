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

# initialising start time
startTime = time.perf_counter()
print("\nStarting...")

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
Tset = [36, 72, 108, 144, 180, 225, 270, 315, 360, 450, 540, 600, 660, 720]
ES = len(Tset)

# list of the lambdas and trials that will be explored
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
          1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
trialset = ["Dist", "TAgg", "Trusted", "no_privacy"]
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanValue = np.zeros((TS, ES))
meanEstMSE = np.zeros((TS, ES))
meanLdaOpt = np.zeros((TS, ES))
meanEstZero = np.zeros((TS, ES))
meanEstOne = np.zeros((TS, ES))
meanPerc = np.zeros((TS, ES))
meanLdaZeroBound = np.zeros((TS, ES))
meanTSmall = np.zeros((TS, LS))
meanTDef = np.zeros((TS, LS))
meanTMid = np.zeros((TS, LS))
meanTLarge = np.zeros((TS, LS))

meanEstRange = np.zeros((TS, ES))
meanLdaOptRange = np.zeros((TS, ES))
meanEstZeroRange = np.zeros((TS, ES))
meanEstOneRange = np.zeros((TS, ES))
meanPercRange = np.zeros((TS, ES))
meanTSmallRange = np.zeros((TS, LS))
meanTDefRange = np.zeros((TS, LS))
meanTMidRange = np.zeros((TS, LS))
meanTLargeRange = np.zeros((TS, LS))

# related to min pairs
minValue = np.zeros((TS, ES))
minEstMSE = np.zeros((TS, ES))
minLdaOpt = np.zeros((TS, ES))
minEstZero = np.zeros((TS, ES))
minEstOne = np.zeros((TS, ES))
minPerc = np.zeros((TS, ES))
minTSmall = np.zeros((TS, LS))
minTDef = np.zeros((TS, LS))
minTMid = np.zeros((TS, LS))
minTLarge = np.zeros((TS, LS))

minEstRange = np.zeros((TS, ES))
minLdaOptRange = np.zeros((TS, ES))
minEstZeroRange = np.zeros((TS, ES))
minEstOneRange = np.zeros((TS, ES))
minPercRange = np.zeros((TS, ES))
minTSmallRange = np.zeros((TS, LS))
minTDefRange = np.zeros((TS, LS))
minTMidRange = np.zeros((TS, LS))
minTLargeRange = np.zeros((TS, LS))

# related to max pairs
maxValue = np.zeros((TS, ES))
maxEstMSE = np.zeros((TS, ES))
maxLdaOpt = np.zeros((TS, ES))
maxEstZero = np.zeros((TS, ES))
maxEstOne = np.zeros((TS, ES))
maxPerc = np.zeros((TS, ES))
maxTSmall = np.zeros((TS, LS))
maxTDef = np.zeros((TS, LS))
maxTMid = np.zeros((TS, LS))
maxTLarge = np.zeros((TS, LS))

maxEstRange = np.zeros((TS, ES))
maxLdaOptRange = np.zeros((TS, ES))
maxEstZeroRange = np.zeros((TS, ES))
maxEstOneRange = np.zeros((TS, ES))
maxPercRange = np.zeros((TS, ES))
maxTSmallRange = np.zeros((TS, LS))
maxTDefRange = np.zeros((TS, LS))
maxTMidRange = np.zeros((TS, LS))
maxTLargeRange = np.zeros((TS, LS))

# global parameters
ALPHA = 0.01 # smoothing parameter
E = 17 # size of subset for k3 estimator
EPS = 0.5
DTA = 0.1
A = 0 # parameter for addition of noise
R1 = 90
ldaStep = 0.05
RS = 10
SEED_FREQ = 0
SMALL_INDEX = 0
DEF_INDEX = 4
MID_INDEX = 7
LARGE_INDEX = 10

for trial in range(4):
    meanfile = open(f"Data_{trialset[trial]}_T_mean.txt", "w", encoding = 'utf-8')
    minfile = open(f"Data_{trialset[trial]}_T_min.txt", "w", encoding = 'utf-8')
    maxfile = open(f"Data_{trialset[trial]}_T_max.txt", "w", encoding = 'utf-8')
    T_FREQ = 0

    for T in Tset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        # temporary stores for each repeat
        tempMeanValue = np.zeros(RS)
        tempMeanInvValue = np.zeros(RS)
        tempMeanRatio = np.zeros(RS)
        tempMeanInvRatio = np.zeros(RS)
        tempMeanEst = np.zeros(RS)
        tempMeanInvEst  = np.zeros(RS)
        tempMeanEstMSE = np.zeros(RS)
        tempMeanLdaOpt = np.zeros(RS)
        tempMeanEstZero = np.zeros(RS)
        tempMeanEstOne = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        tempMeanTSmall = np.zeros((LS, RS))
        tempMeanTDef = np.zeros((LS, RS))
        tempMeanTMid = np.zeros((LS, RS))
        tempMeanTLarge = np.zeros((LS, RS))
            
        tempMinValue = np.zeros(RS)
        tempMinInvValue = np.zeros(RS)
        tempMinRatio = np.zeros(RS)
        tempMinInvRatio = np.zeros(RS)
        tempMinEst = np.zeros(RS)
        tempMinInvEst = np.zeros(RS)
        tempMinEstMSE = np.zeros(RS)
        tempMinLdaOpt = np.zeros(RS)
        tempMinEstZero = np.zeros(RS)
        tempMinEstOne = np.zeros(RS)
        tempMinPerc = np.zeros(RS)
        tempMinTSmall = np.zeros((LS, RS))
        tempMinTDef = np.zeros((LS, RS))
        tempMinTMid = np.zeros((LS, RS))
        tempMinTLarge = np.zeros((LS, RS))

        tempMaxValue = np.zeros(RS)
        tempMaxInvValue = np.zeros(RS)
        tempMaxRatio = np.zeros(RS)
        tempMaxInvRatio = np.zeros(RS)
        tempMaxEst = np.zeros(RS)
        tempMaxInvEst = np.zeros(RS)
        tempMaxEstMSE = np.zeros(RS)
        tempMaxLdaOpt = np.zeros(RS)
        tempMaxEstZero = np.zeros(RS)
        tempMaxEstOne = np.zeros(RS)
        tempMaxPerc = np.zeros(RS)
        tempMaxTSmall = np.zeros((LS, RS))
        tempMaxTDef = np.zeros((LS, RS))
        tempMaxTMid = np.zeros((LS, RS))
        tempMaxTLarge = np.zeros((LS, RS))

        tempVarNoise = np.zeros(RS)

        for rep in range(RS):      
            print(f"T = {T}, repeat {rep + 1}...")

            # initialising seeds for random sampling
            tf.random.set_seed(SEED_FREQ)
            np.random.seed(SEED_FREQ)

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
            T1 = 11*T

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

            # load Gaussian noise distributions for clients and intermediate server
            b1 = (1 + log(2)) / EPS
            b2 = (2*((log(1.25))/DTA)*b1) / EPS

            if trial < 2:
                s = b2 * (np.sqrt(2) / R1)
                probGaussNoise = tfp.distributions.Normal(loc = A, scale = s / 100)
                gaussNoise = tfp.distributions.Normal(loc = A, scale = s)

            # stores for exact unknown distributions
            uDist = np.zeros((10, 10, U))
            nDist = np.zeros((10, 10, E))
            uList = []
            nList = []
            uCDList = []         
            rList = []
            rInvList = []
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
                    
                    # eliminate all zero values when digits are identical
                    if sum(nDist[C, D]) != 0.0:
                        nList.append(sum(nDist[C, D]))

                    # compute ratio (and its inverse) between exact unknown distributions
                    ratio = abs(sum(nDist[C, D]) / sum(uDist[C, D]))
                    invRatio = abs(sum(uDist[C, D]) / sum(nDist[C, D]))

                    # eliminate all divide by zero errors
                    if ratio != 0.0 and sum(uDist[C, D]) != 0.0:

                        # "Dist" (each client adds Gaussian noise term)
                        if trial == 0:
                            startSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                            startNoise.append(startSample)
                            ratio = ratio + startSample
                            invRatio = invRatio + startSample
                    
                        rList.append(ratio)
                        rInvList.append(ratio)

            # store for PRIEST-KLD
            R2 = len(rList)
            uEst = np.zeros((LS, R2))
            nEst = np.zeros((LS, R2))
            R_FREQ = 0

            for row in range(0, R2):
                uLogr = np.log(rList[row])
                nLogr = np.log(rInvList[row])
                LDA_FREQ = 0

                # explore lambdas in a range
                for lda in ldaset:

                    # compute k3 estimator (and its inverse)
                    uRangeEst = lda * (np.exp(uLogr) - 1) - uLogr
                    nRangeEst = lda * (np.exp(nLogr) - 1) - nLogr

                    # share PRIEST-KLD with intermediate server
                    uEst[LDA_FREQ, R_FREQ] = uRangeEst
                    nEst[LDA_FREQ, R_FREQ] = nRangeEst
                    LDA_FREQ = LDA_FREQ + 1

                R_FREQ = R_FREQ + 1
        
            # extract position and identity of max and min pairs
            minIndex = np.argmin(uList)
            maxIndex = np.argmax(uList)
            minInvIndex = np.argmin(nList)
            maxInvIndex = np.argmax(nList)

            minPair = uCDList[minIndex]
            maxPair = uCDList[maxIndex]

            # extract ground truths
            tempMeanValue[rep] = np.mean(uList)
            tempMeanInvValue[rep] = np.mean(nList)

            tempMinValue[rep] = uList[minIndex]
            tempMaxValue[rep] = uList[maxIndex]
            tempMinInvValue[rep] = nList[minInvIndex]
            tempMaxInvValue[rep] = nList[maxInvIndex]

            # extract mean / min / max ratios (and inverses) for Theorem 4.4 and Corollary 4.5
            tempMeanRatio[rep] = np.mean(rList)
            tempMeanInvRatio[rep] = np.mean(rInvList)

            tempMinRatio[rep] = rList[minIndex]
            tempMaxRatio[rep] = rList[maxIndex]            
            tempMinInvRatio[rep] = rInvList[minInvIndex]
            tempMaxInvRatio[rep] = rInvList[maxInvIndex]

            meanLda = np.zeros(LS)
            minLda = np.zeros(LS)
            maxLda = np.zeros(LS)
            meanInvLda = np.zeros(LS)
            minInvLda = np.zeros(LS)
            maxInvLda = np.zeros(LS)

            meanLdaNoise = np.zeros(LS)
            minLdaNoise = np.zeros(LS)
            maxLdaNoise = np.zeros(LS)
            meanInvLdaNoise = np.zeros(LS)
            minInvLdaNoise = np.zeros(LS)
            maxInvLdaNoise = np.zeros(LS)

            # compute mean error of PRIEST-KLD for each lambda
            for l in range(LS):
                meanLda[l] = np.mean(uEst[l])
                meanInvLda[l] = np.mean(nEst[l])

                # extract error for max and min pairs
                minLda[l] = uEst[l, minIndex]
                maxLda[l] = uEst[l, maxIndex]
                minInvLda[l] = nEst[l, minIndex]
                maxInvLda[l] = nEst[l, maxIndex]

                # "TAgg" (intermediate server adds Gaussian noise term)
                if trial == 1:
                    meanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    minLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    maxLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))

                    meanLda[l] = meanLda[l] + meanLdaNoise[l]
                    minLda[l] = minLda[l] + minLdaNoise[l]
                    maxLda[l] = maxLda[l] + maxLdaNoise[l]
                    meanInvLda[l] = meanInvLda[l] + meanLdaNoise[l]
                    minInvLda[l] = minInvLda[l] + minLdaNoise[l]
                    maxInvLda[l] = maxInvLda[l] + maxLdaNoise[l]
            
                # mean / min / max across lambdas for T = 36 (~1% of clients, small)
                if T_FREQ == SMALL_INDEX:
                    tempMeanTSmall[l, rep] = meanLda[l]
                    tempMinTSmall[l, rep] = minLda[l]
                    tempMaxTSmall[l, rep] = maxLda[l]

                # T = 180 (~5% of clients, default)
                if T_FREQ == DEF_INDEX:
                    tempMeanTDef[l, rep] = meanLda[l]
                    tempMinTDef[l, rep] = minLda[l]
                    tempMaxTDef[l, rep] = maxLda[l]

                # T = 360 (~10% of clients, mid)
                if T_FREQ == MID_INDEX:
                    tempMeanTMid[l, rep] = meanLda[l]
                    tempMinTMid[l, rep] = minLda[l]
                    tempMaxTMid[l, rep] = maxLda[l]

                # T = 600 (~16.6% of clients, large)
                if T_FREQ == LARGE_INDEX:
                    tempMeanTLarge[l, rep] = meanLda[l]
                    tempMinTLarge[l, rep] = minLda[l]
                    tempMaxTLarge[l, rep] = maxLda[l]

            # find lambda that produces minimum error
            meanLdaIndex = np.argmin(meanLda)
            minLdaIndex = np.argmin(minLda)
            maxLdaIndex = np.argmin(maxLda)
            meanInvLdaIndex = np.argmin(meanInvLda)
            minInvLdaIndex = np.argmin(minInvLda)
            maxInvLdaIndex = np.argmin(maxInvLda)

            meanMinError = meanLda[meanLdaIndex]
            minMinError = minLda[minLdaIndex]
            maxMinError = maxLda[maxLdaIndex]
            meanInvMinError = meanInvLda[meanInvLdaIndex]
            minInvMinError = minInvLda[minInvLdaIndex]
            maxInvMinError = maxLda[maxInvLdaIndex]

            # mean / min / max across clients for optimum lambda
            tempMeanEst[rep] = meanMinError
            tempMinEst[rep] = minMinError
            tempMaxEst[rep] = maxMinError
            tempMeanInvEst[rep] = meanInvMinError
            tempMinInvEst[rep] = minInvMinError
            tempMaxInvEst[rep] = maxInvMinError

            # optimum lambda
            tempMeanLdaOpt[rep] = meanLdaIndex * ldaStep
            tempMinLdaOpt[rep] = minLdaIndex * ldaStep
            tempMaxLdaOpt[rep] = maxLdaIndex * ldaStep

            # lambda = 0
            tempMeanEstZero[rep] = meanLda[0]
            tempMinEstZero[rep] = minLda[0]
            tempMaxEstZero[rep] = maxLda[0]

            # lambda = 1
            tempMeanEstOne[rep] = meanLda[LS-1]
            tempMinEstOne[rep] = minLda[LS-1]
            tempMaxEstOne[rep] = maxLda[LS-1]

            # "Trusted" (server adds Laplace noise term to final result)
            if trial == 2:
                lapNoise = tfp.distributions.Normal(loc = A, scale = b2)
                meanNoise = lapNoise.sample(sample_shape = (1,))
                minNoise = lapNoise.sample(sample_shape = (1,))
                maxNoise = lapNoise.sample(sample_shape = (1,))
                meanZeroNoise = lapNoise.sample(sample_shape = (1,))
                minZeroNoise = lapNoise.sample(sample_shape = (1,))
                maxZeroNoise = lapNoise.sample(sample_shape = (1,))
                meanOneNoise = lapNoise.sample(sample_shape = (1,))
                minOneNoise = lapNoise.sample(sample_shape = (1,))
                maxOneNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEstMSE[rep] = (tempMeanEst[rep] + meanNoise - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] + minNoise - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] + maxNoise - tempMaxValue[rep])**2

                # lambda = 0
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] + meanZeroNoise - tempMeanValue[rep])**2
                tempMinEstZero[rep] = (tempMinEstZero[rep] + minZeroNoise - tempMinValue[rep])**2
                tempMaxEstZero[rep] = (tempMaxEstZero[rep] + maxZeroNoise - tempMaxValue[rep])**2

                # lambda = 1
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] + meanOneNoise - tempMeanValue[rep])**2
                tempMinEstOne[rep] = (tempMinEstOne[rep] + minOneNoise - tempMinValue[rep])**2
                tempMaxEstOne[rep] = (tempMaxEstOne[rep] + maxOneNoise - tempMaxValue[rep])**2

                for l in range(LS):
        
                    # T = 36 (small)
                    if T_FREQ == SMALL_INDEX:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        minSmallNoise = lapNoise.sample(sample_shape = (1,))
                        maxSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanTSmall[l, rep] = (tempMeanTSmall[l, rep] + meanSmallNoise - tempMeanValue[rep])**2
                        tempMinTSmall[l, rep] = (tempMinTSmall[l, rep] + minSmallNoise - tempMinValue[rep])**2
                        tempMaxTSmall[l, rep] = (tempMaxTSmall[l, rep] + maxSmallNoise - tempMaxValue[rep])**2

                    # T = 180 (def)
                    if T_FREQ == DEF_INDEX:
                        meanDefNoise = lapNoise.sample(sample_shape = (1,))
                        minDefNoise = lapNoise.sample(sample_shape = (1,))
                        maxDefNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanTDef[l, rep] = (tempMeanTDef[l, rep] + meanDefNoise - tempMeanValue[rep])**2
                        tempMinTDef[l, rep] = (tempMinTDef[l, rep] + minDefNoise - tempMinValue[rep])**2
                        tempMaxTDef[l, rep] = (tempMaxTDef[l, rep] + maxDefNoise - tempMaxValue[rep])**2

                    # T = 360 (mid)
                    if T_FREQ == MID_INDEX:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        minMidNoise = lapNoise.sample(sample_shape = (1,))
                        maxMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanTMid[l, rep] = (tempMeanTMid[l, rep] + meanMidNoise - tempMeanValue[rep])**2
                        tempMinTMid[l, rep] = (tempMinTMid[l, rep] + minMidNoise - tempMinValue[rep])**2
                        tempMaxTMid[l, rep] = (tempMaxTMid[l, rep] + maxMidNoise - tempMaxValue[rep])**2

                    # T = 600 (large)
                    if T_FREQ == LARGE_INDEX:
                        meanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        minLargeNoise = lapNoise.sample(sample_shape = (1,))
                        maxLargeNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanTLarge[l, rep] = (tempMeanTLarge[l, rep] + meanLargeNoise - tempMeanValue[rep])**2
                        tempMinTLarge[l, rep] = (tempMinTLarge[l, rep] + minLargeNoise - tempMinValue[rep])**2
                        tempMaxTLarge[l, rep] = (tempMaxTLarge[l, rep] + maxLargeNoise - tempMaxValue[rep])**2
        
            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEstMSE[rep] = (tempMeanEst[rep] - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] - tempMaxValue[rep])**2

                # lambda = 0
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] - tempMeanValue[rep])**2
                tempMinEstZero[rep] = (tempMinEstZero[rep] - tempMinValue[rep])**2
                tempMaxEstZero[rep] = (tempMaxEstZero[rep] - tempMaxValue[rep])**2

                # lambda = 1
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] - tempMeanValue[rep])**2
                tempMinEstOne[rep] = (tempMinEstOne[rep] - tempMinValue[rep])**2
                tempMaxEstOne[rep] = (tempMaxEstOne[rep] - tempMaxValue[rep])**2

                for l in range(LS):

                    # T = 36 (small)
                    if T_FREQ == SMALL_INDEX:
                        tempMeanTSmall[l, rep] = (tempMeanTSmall[l, rep] - tempMeanValue[rep])**2
                        tempMinTSmall[l, rep] = (tempMinTSmall[l, rep] - tempMinValue[rep])**2
                        tempMaxTSmall[l, rep] = (tempMaxTSmall[l, rep] - tempMaxValue[rep])**2

                    # T = 180 (def)
                    if T_FREQ == DEF_INDEX:
                        tempMeanTDef[l, rep] = (tempMeanTDef[l, rep] - tempMeanValue[rep])**2
                        tempMinTDef[l, rep] = (tempMinTDef[l, rep] - tempMinValue[rep])**2
                        tempMaxTDef[l, rep] = (tempMaxTDef[l, rep] - tempMaxValue[rep])**2

                    # T = 360 (mid)
                    if T_FREQ == MID_INDEX:
                        tempMeanTMid[l, rep] = (tempMeanTMid[l, rep] - tempMeanValue[rep])**2
                        tempMinTMid[l, rep] = (tempMinTMid[l, rep] - tempMinValue[rep])**2
                        tempMaxTMid[l, rep] = (tempMaxTMid[l, rep] - tempMaxValue[rep])**2

                    # T = 600 (large)
                    if T_FREQ == LARGE_INDEX:
                        tempMeanTLarge[l, rep] = (tempMeanTLarge[l, rep] - tempMeanValue[rep])**2
                        tempMinTLarge[l, rep] = (tempMinTLarge[l, rep] - tempMinValue[rep])**2
                        tempMaxTLarge[l, rep] = (tempMaxTLarge[l, rep] - tempMaxValue[rep])**2

            # compute % of noise vs ground truth
            if trial == 0:
                tempMeanPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMeanValue[rep])))*100
                tempMinPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMinValue[rep])))*100
                tempMaxPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMaxValue[rep])))*100
                tempVarNoise[rep] = np.max(startNoise)       

            if trial == 1:
                tempMeanPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMeanValue[rep]))*100
                tempMinPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMinValue[rep]))*100
                tempMaxPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMaxValue[rep]))*100
                tempVarNoise[rep] = np.max(meanLdaNoise)
            
            if trial == 2:
                tempMeanPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMeanValue[rep])))*100
                tempMinPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMinValue[rep])))*100
                tempMaxPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMaxValue[rep])))*100
                tempVarNoise[rep] = np.max(meanNoise)
        
            SEED_FREQ = SEED_FREQ + 1

        # compute mean of repeats
        meanValue[trial, T_FREQ] = np.mean(tempMeanValue)
        meanEstMSE[trial, T_FREQ] = np.mean(tempMeanEstMSE)
        meanLdaOpt[trial, T_FREQ] = np.mean(tempMeanLdaOpt)
        meanEstZero[trial, T_FREQ] = np.mean(tempMeanEstZero)
        meanEstOne[trial, T_FREQ] = np.mean(tempMeanEstOne)
        meanPerc[trial, T_FREQ] = np.mean(tempMeanPerc)
        meanEst = np.mean(tempMeanEst)
        meanInvEst = np.mean(tempMeanInvEst)

        for l in range(LS):
            if T_FREQ == SMALL_INDEX:
                meanTSmall[trial, l] = np.mean(tempMeanTSmall[l])
            if T_FREQ == DEF_INDEX:
                meanTDef[trial, l] = np.mean(tempMeanTDef[l])
            if T_FREQ == MID_INDEX:
                meanTMid[trial, l] = np.mean(tempMeanTMid[l])
            if T_FREQ == LARGE_INDEX:
                meanTLarge[trial, l] = np.mean(tempMeanTLarge[l])

        minValue[trial, T_FREQ] = np.mean(tempMinValue)
        minEstMSE[trial, T_FREQ] = np.mean(tempMinEstMSE)
        minLdaOpt[trial, T_FREQ] = np.mean(tempMinLdaOpt)
        minEstZero[trial, T_FREQ] = np.mean(tempMinEstZero)
        minEstOne[trial, T_FREQ] = np.mean(tempMinEstOne)
        minPerc[trial, T_FREQ] = np.mean(tempMinPerc)
        minEst = np.mean(tempMinEst)
        minInvEst = np.mean(tempMinInvEst)

        for l in range(LS):
            if T_FREQ == SMALL_INDEX:
                minTSmall[trial, l] = np.mean(tempMinTSmall[l])
            if T_FREQ == DEF_INDEX:
                minTDef[trial, l] = np.mean(tempMinTDef[l])
            if T_FREQ == MID_INDEX:
                minTMid[trial, l] = np.mean(tempMinTMid[l])
            if T_FREQ == LARGE_INDEX:
                minTLarge[trial, l] = np.mean(tempMinTLarge[l])

        maxValue[trial, T_FREQ] = np.mean(tempMaxValue)
        maxEstMSE[trial, T_FREQ] = np.mean(tempMaxEstMSE)
        maxLdaOpt[trial, T_FREQ] = np.mean(tempMaxLdaOpt)
        maxEstZero[trial, T_FREQ] = np.mean(tempMaxEstZero)
        maxEstOne[trial, T_FREQ] = np.mean(tempMaxEstOne)
        maxPerc[trial, T_FREQ] = np.mean(tempMaxPerc)
        maxEst = np.mean(tempMaxEst)
        maxInvEst = np.mean(tempMaxInvEst)

        for l in range(LS):
            if T_FREQ == SMALL_INDEX:
                maxTSmall[trial, l] = np.mean(tempMaxTSmall[l])
            if T_FREQ == DEF_INDEX:
                maxTDef[trial, l] = np.mean(tempMaxTDef[l])
            if T_FREQ == MID_INDEX:
                maxTMid[trial, l] = np.mean(tempMaxTMid[l])
            if T_FREQ == LARGE_INDEX:
                maxTLarge[trial, l] = np.mean(tempMaxTLarge[l])

        # compute standard deviation of repeats
        meanEstRange[trial, T_FREQ] = np.std(tempMeanEstMSE)
        meanLdaOptRange[trial, T_FREQ] = np.std(tempMeanLdaOpt)
        meanEstZeroRange[trial, T_FREQ] = np.std(tempMeanEstZero)
        meanEstOneRange[trial, T_FREQ] = np.std(tempMeanEstOne)
        meanPercRange[trial, T_FREQ] = np.std(tempMeanPerc)

        for l in range(LS):
            if T_FREQ == SMALL_INDEX:
                meanTSmallRange[trial, l] = np.std(tempMeanTSmall[l])
            if T_FREQ == DEF_INDEX:
                meanTDefRange[trial, l] = np.std(tempMeanTDef[l])
            if T_FREQ == MID_INDEX:
                meanTMidRange[trial, l] = np.std(tempMeanTMid[l])
            if T_FREQ == LARGE_INDEX:
                meanTLargeRange[trial, l] = np.std(tempMeanTLarge[l])

        minEstRange[trial, T_FREQ] = np.std(tempMinEstMSE)
        minLdaOptRange[trial, T_FREQ] = np.std(tempMinLdaOpt)
        minEstZeroRange[trial, T_FREQ] = np.std(tempMinEstZero)
        minEstOneRange[trial, T_FREQ] = np.std(tempMinEstOne)
        minPercRange[trial, T_FREQ] = np.std(tempMinPerc)

        for l in range(LS):
            if T_FREQ == SMALL_INDEX:
                minTSmallRange[trial, l] = np.std(tempMinTSmall[l])
            if T_FREQ == DEF_INDEX:
                minTDefRange[trial, l] = np.std(tempMinTDef[l])
            if T_FREQ == MID_INDEX:
                minTMidRange[trial, l] = np.std(tempMinTMid[l])
            if T_FREQ == LARGE_INDEX:
                minTLargeRange[trial, l] = np.std(tempMinTLarge[l])

        maxEstRange[trial, T_FREQ] = np.std(tempMaxEstMSE)
        maxLdaOptRange[trial, T_FREQ] = np.std(tempMaxLdaOpt)
        maxEstZeroRange[trial, T_FREQ] = np.std(tempMaxEstZero)
        maxEstOneRange[trial, T_FREQ] = np.std(tempMaxEstOne)
        maxPercRange[trial, T_FREQ] = np.std(tempMaxPerc)

        for l in range(LS):
            if T_FREQ == SMALL_INDEX:
                maxTSmallRange[trial, l] = np.std(tempMaxTSmall[l])
            if T_FREQ == DEF_INDEX:
                maxTDefRange[trial, l] = np.std(tempMaxTDef[l])
            if T_FREQ == MID_INDEX:
                maxTMidRange[trial, l] = np.std(tempMaxTMid[l])
            if T_FREQ == LARGE_INDEX:
                maxTLargeRange[trial, l] = np.std(tempMaxTLarge[l])

        # compute alpha and beta for Theorem 4.4 and Corollary 4.5 using mean / min / max ratios
        meanAlpha = np.max(tempMeanRatio)
        minAlpha = np.max(tempMinRatio)
        maxAlpha = np.max(tempMaxRatio)
        meanBeta = np.max(tempMeanInvRatio)
        minBeta = np.max(tempMinInvRatio)
        maxBeta = np.max(tempMaxInvRatio)
        varNoise = np.var(tempVarNoise)
        meanLdaBound = 0
        minLdaBound = 0
        maxLdaBound = 0
        meanMaxLda = 0
        minMaxLda = 0
        maxMaxLda = 0

        # find maximum lambda that satisfies MSE upper bound in Theorem 4.4
        for lda in ldaset:
            if meanLdaBound < meanEstMSE[trial, T_FREQ]:
                meanLdaBound = ((lda**2 / T) * (meanAlpha - 1)) + ((1 / T) * (max(meanAlpha - 1, meanBeta**2 - 1)
                    + meanEst**2)) - ((2*lda / T) * (meanInvEst + meanEst)) + varNoise
                meanMaxLda = lda
            
            if minLdaBound < minEstMSE[trial, T_FREQ]:
                minLdaBound = ((lda**2 / T) * (minAlpha - 1)) + ((1 / T) * (max(minAlpha - 1, minBeta**2 - 1)
                    + minEst**2)) - ((2*lda / T) * (minInvEst + minEst)) + varNoise
                minMaxLda = lda
            
            if maxLdaBound < maxEstMSE[trial, T_FREQ]:
                maxLdaBound = ((lda**2 / T) * (maxAlpha - 1)) + ((1 / T) * (max(maxAlpha - 1, maxBeta**2 - 1)
                    + maxEst**2)) - ((2*lda / T) * (maxInvEst + maxEst)) + varNoise
                maxMaxLda = lda

        # compute optimal lambda upper bound in Corollary 4.5
        meanLdaOptBound = ((meanAlpha * meanBeta) - 1)**2 / (2 * meanAlpha * meanBeta * (meanAlpha - 1))
        minLdaOptBound = ((minAlpha * minBeta) - 1)**2 / (2 * minAlpha * minBeta * (minAlpha - 1))
        maxLdaOptBound = ((maxAlpha * maxBeta) - 1)**2 / (2 * maxAlpha * maxBeta * (maxAlpha - 1))

        # write statistics on data files
        if T == Tset[0]:
            meanfile.write(f"FEMNIST: T = {T}\n")
            minfile.write(f"FEMNIST: T = {T}\n")
            maxfile.write(f"FEMNIST: T = {T}\n")
        else:
            meanfile.write(f"\nT = {T}\n")
            minfile.write(f"\nT = {T}\n")
            maxfile.write(f"\nT = {T}\n")
  
        meanfile.write(f"\nMean MSE: {round(meanEstMSE[trial, T_FREQ], 2)}\n")
        meanfile.write(f"MSE Upper Bound: {round(meanLdaBound, 2)}\n")
        meanfile.write(f"Maximum Lambda: {round(meanMaxLda, 2)}\n")
        meanfile.write(f"Optimal Lambda: {round(meanLdaOpt[trial, T_FREQ], 2)}\n")
        meanfile.write(f"Optimal Lambda Upper Bound: {round(meanLdaOptBound, 2)}\n")
        meanfile.write(f"Ground Truth: {round(meanValue[trial, T_FREQ], 2)}\n")
        meanfile.write(f"Noise: {np.round(meanPerc[trial, T_FREQ], 2)}%\n")

        minfile.write(f"\nMin Pair: {minPair}\n")
        minfile.write(f"Min MSE: {round(minEstMSE[trial, T_FREQ], 2)}\n")
        minfile.write(f"MSE Upper Bound: {round(minLdaBound, 2)}\n")
        minfile.write(f"Maximum Lambda: {round(minMaxLda, 2)}\n")
        minfile.write(f"Optimal Lambda: {round(minLdaOpt[trial, T_FREQ], 2)}\n")
        minfile.write(f"Optimal Lambda Upper Bound: {round(minLdaOptBound, 2)}\n")
        minfile.write(f"Ground Truth: {round(minValue[trial, T_FREQ], 2)}\n")
        minfile.write(f"Noise: {np.round(minPerc[trial, T_FREQ], 2)}%\n")

        minfile.write(f"\nMax Pair: {maxPair}\n")
        maxfile.write(f"Max MSE: {round(maxEstMSE[trial, T_FREQ], 2)}\n")
        maxfile.write(f"MSE Upper Bound: {round(maxLdaBound, 2)}\n")
        maxfile.write(f"Maximum Lambda: {round(maxMaxLda, 2)}\n")
        maxfile.write(f"Optimal Lambda: {round(maxLdaOpt[trial, T_FREQ], 2)}\n")
        maxfile.write(f"Optimal Lambda Upper Bound: {round(maxLdaOptBound, 2)}\n")
        maxfile.write(f"Ground Truth: {round(maxValue[trial, T_FREQ], 2)}\n")
        maxfile.write(f"Noise: {np.round(maxPerc[trial, T_FREQ], 2)}%\n")

        T_FREQ = T_FREQ + 1

# EXPERIMENT 1: MSE of PRIEST-KLD for each T
plt.errorbar(Tset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanEstMSE[3], yerr = np.minimum(meanEstRange[3], np.sqrt(meanEstMSE[3]), np.divide(meanEstMSE[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_T_est_mean.png")
plt.clf()

plt.errorbar(Tset, minEstMSE[0], yerr = np.minimum(minEstRange[0], np.sqrt(minEstMSE[0]), np.divide(minEstMSE[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minEstMSE[1], yerr = np.minimum(minEstRange[1], np.sqrt(minEstMSE[1]), np.divide(minEstMSE[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minEstMSE[2], yerr = np.minimum(minEstRange[2], np.sqrt(minEstMSE[2]), np.divide(minEstMSE[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minEstMSE[3], yerr = np.minimum(minEstRange[3], np.sqrt(minEstMSE[3]), np.divide(minEstMSE[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_T_est_min.png")
plt.clf()

plt.errorbar(Tset, maxEstMSE[0], yerr = np.minimum(maxEstRange[0], np.sqrt(maxEstMSE[0]), np.divide(maxEstMSE[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxEstMSE[1], yerr = np.minimum(maxEstRange[1], np.sqrt(maxEstMSE[1]), np.divide(maxEstMSE[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxEstMSE[2], yerr = np.minimum(maxEstRange[2], np.sqrt(maxEstMSE[2]), np.divide(maxEstMSE[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxEstMSE[3], yerr = np.minimum(maxEstRange[3], np.sqrt(maxEstMSE[3]), np.divide(maxEstMSE[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_T_est_max.png")
plt.clf()

# EXPERIMENT 2: MSE of PRIEST-KLD when lambda = 0 for each T
plt.errorbar(Tset, meanEstZero[0], yerr = np.minimum(meanEstZeroRange[0], np.sqrt(meanEstZero[0]), np.divide(meanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanEstZero[1], yerr = np.minimum(meanEstZeroRange[1], np.sqrt(meanEstZero[1]), np.divide(meanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanEstZero[2], yerr = np.minimum(meanEstZeroRange[2], np.sqrt(meanEstZero[2]), np.divide(meanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanEstZero[3], yerr = np.minimum(meanEstZeroRange[3], np.sqrt(meanEstZero[3]), np.divide(meanEstZero[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_lda_zero_mean.png")
plt.clf()

plt.errorbar(Tset, minEstZero[0], yerr = np.minimum(minEstZeroRange[0], np.sqrt(minEstZero[0]), np.divide(minEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minEstZero[1], yerr = np.minimum(minEstZeroRange[1], np.sqrt(minEstZero[1]), np.divide(minEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minEstZero[2], yerr = np.minimum(minEstZeroRange[2], np.sqrt(minEstZero[2]), np.divide(minEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minEstZero[3], yerr = np.minimum(minEstZeroRange[3], np.sqrt(minEstZero[3]), np.divide(minEstZero[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_lda_zero_min.png")
plt.clf()

plt.errorbar(Tset, maxEstZero[0], yerr = np.minimum(maxEstZeroRange[0], np.sqrt(maxEstZero[0]), np.divide(maxEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxEstZero[1], yerr = np.minimum(maxEstZeroRange[1], np.sqrt(maxEstZero[1]), np.divide(maxEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxEstZero[2], yerr = np.minimum(maxEstZeroRange[2], np.sqrt(maxEstZero[2]), np.divide(maxEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxEstZero[3], yerr = np.minimum(maxEstZeroRange[3], np.sqrt(maxEstZero[3]), np.divide(maxEstZero[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_lda_zero_max.png")
plt.clf()

# EXPERIMENT 2: MSE of PRIEST-KLD when lambda = 1 for each T
plt.errorbar(Tset, meanEstOne[0], yerr = np.minimum(meanEstOneRange[0], np.sqrt(meanEstOne[0]), np.divide(meanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanEstOne[1], yerr = np.minimum(meanEstOneRange[1], np.sqrt(meanEstOne[1]), np.divide(meanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanEstOne[2], yerr = np.minimum(meanEstOneRange[2], np.sqrt(meanEstOne[2]), np.divide(meanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanEstOne[3], yerr = np.minimum(meanEstOneRange[3], np.sqrt(meanEstOne[3]), np.divide(meanEstOne[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_lda_one_mean.png")
plt.clf()

plt.errorbar(Tset, minEstOne[0], yerr = np.minimum(minEstOneRange[0], np.sqrt(minEstOne[0]), np.divide(minEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minEstOne[1], yerr = np.minimum(minEstOneRange[1], np.sqrt(minEstOne[1]), np.divide(minEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minEstOne[2], yerr = np.minimum(minEstOneRange[2], np.sqrt(minEstOne[2]), np.divide(minEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minEstOne[3], yerr = np.minimum(minEstOneRange[3], np.sqrt(minEstOne[3]), np.divide(minEstOne[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_lda_one_min.png")
plt.clf()

plt.errorbar(Tset, maxEstOne[0], yerr = np.minimum(maxEstOneRange[0], np.sqrt(maxEstOne[0]), np.divide(maxEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxEstOne[1], yerr = np.minimum(maxEstOneRange[1], np.sqrt(maxEstOne[1]), np.divide(maxEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxEstOne[2], yerr = np.minimum(maxEstOneRange[2], np.sqrt(maxEstOne[2]), np.divide(maxEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxEstOne[3], yerr = np.minimum(maxEstOneRange[3], np.sqrt(maxEstOne[3]), np.divide(maxEstOne[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_lda_one_max.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD when T = 36
plt.errorbar(ldaset, meanTSmall[0], yerr = np.minimum(meanTSmallRange[0], np.sqrt(meanTSmall[0]), np.divide(meanTSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTSmall[1], yerr = np.minimum(meanTSmallRange[1], np.sqrt(meanTSmall[1]), np.divide(meanTSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTSmall[2], yerr = np.minimum(meanTSmallRange[2], np.sqrt(meanTSmall[2]), np.divide(meanTSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTSmall[3], yerr = np.minimum(meanTSmallRange[3], np.sqrt(meanTSmall[3]), np.divide(meanTSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_small_mean.png")
plt.clf()

plt.errorbar(ldaset, minTSmall[0], yerr = np.minimum(minTSmallRange[0], np.sqrt(minTSmall[0]), np.divide(minTSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTSmall[1], yerr = np.minimum(minTSmallRange[1], np.sqrt(minTSmall[1]), np.divide(minTSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTSmall[2], yerr = np.minimum(minTSmallRange[2], np.sqrt(minTSmall[2]), np.divide(minTSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTSmall[3], yerr = np.minimum(minTSmallRange[3], np.sqrt(minTSmall[3]), np.divide(minTSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_small_min.png")
plt.clf()

plt.errorbar(ldaset, maxTSmall[0], yerr = np.minimum(maxTSmallRange[0], np.sqrt(maxTSmall[0]), np.divide(maxTSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTSmall[1], yerr = np.minimum(maxTSmallRange[1], np.sqrt(maxTSmall[1]), np.divide(maxTSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTSmall[2], yerr = np.minimum(maxTSmallRange[2], np.sqrt(maxTSmall[2]), np.divide(maxTSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTSmall[3], yerr = np.minimum(maxTSmallRange[3], np.sqrt(maxTSmall[3]), np.divide(maxTSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_small_max.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD when T = 180
plt.errorbar(ldaset, meanTDef[0], yerr = np.minimum(meanTDefRange[0], np.sqrt(meanTDef[0]), np.divide(meanTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTDef[1], yerr = np.minimum(meanTDefRange[1], np.sqrt(meanTDef[1]), np.divide(meanTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTDef[2], yerr = np.minimum(meanTDefRange[2], np.sqrt(meanTDef[2]), np.divide(meanTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTDef[3], yerr = np.minimum(meanTDefRange[3], np.sqrt(meanTDef[3]), np.divide(meanTDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_def_mean.png")
plt.clf()

plt.errorbar(ldaset, minTDef[0], yerr = np.minimum(minTDefRange[0], np.sqrt(minTDef[0]), np.divide(minTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTDef[1], yerr = np.minimum(minTDefRange[1], np.sqrt(minTDef[1]), np.divide(minTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTDef[2], yerr = np.minimum(minTDefRange[2], np.sqrt(minTDef[2]), np.divide(minTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTDef[3], yerr = np.minimum(minTDefRange[3], np.sqrt(minTDef[3]), np.divide(minTDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_def_min.png")
plt.clf()

plt.errorbar(ldaset, maxTDef[0], yerr = np.minimum(maxTDefRange[0], np.sqrt(maxTDef[0]), np.divide(maxTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTDef[1], yerr = np.minimum(maxTDefRange[1], np.sqrt(maxTDef[1]), np.divide(maxTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTDef[2], yerr = np.minimum(maxTDefRange[2], np.sqrt(maxTDef[2]), np.divide(maxTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTDef[3], yerr = np.minimum(maxTDefRange[3], np.sqrt(maxTDef[3]), np.divide(maxTDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_def_max.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD when T = 360
plt.errorbar(ldaset, meanTMid[0], yerr = np.minimum(meanTMidRange[0], np.sqrt(meanTMid[0]), np.divide(meanTMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTMid[1], yerr = np.minimum(meanTMidRange[1], np.sqrt(meanTMid[1]), np.divide(meanTMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTMid[2], yerr = np.minimum(meanTMidRange[2], np.sqrt(meanTMid[2]), np.divide(meanTMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTMid[3], yerr = np.minimum(meanTMidRange[3], np.sqrt(meanTMid[3]), np.divide(meanTMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_mid_mean.png")
plt.clf()

plt.errorbar(ldaset, minTMid[0], yerr = np.minimum(minTMidRange[0], np.sqrt(minTMid[0]), np.divide(minTMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTMid[1], yerr = np.minimum(minTMidRange[1], np.sqrt(minTMid[1]), np.divide(minTMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTMid[2], yerr = np.minimum(minTMidRange[2], np.sqrt(minTMid[2]), np.divide(minTMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTMid[3], yerr = np.minimum(minTMidRange[3], np.sqrt(minTMid[3]), np.divide(minTMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_mid_min.png")
plt.clf()

plt.errorbar(ldaset, maxTMid[0], yerr = np.minimum(maxTMidRange[0], np.sqrt(maxTMid[0]), np.divide(maxTMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTMid[1], yerr = np.minimum(maxTMidRange[1], np.sqrt(maxTMid[1]), np.divide(maxTMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTMid[2], yerr = np.minimum(maxTMidRange[2], np.sqrt(maxTMid[2]), np.divide(maxTMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTMid[3], yerr = np.minimum(maxTMidRange[3], np.sqrt(maxTMid[3]), np.divide(maxTMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_mid_max.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD when T = 600
plt.errorbar(ldaset, meanTLarge[0], yerr = np.minimum(meanTLargeRange[0], np.sqrt(meanTLarge[0]), np.divide(meanTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanTLarge[1], yerr = np.minimum(meanTLargeRange[1], np.sqrt(meanTLarge[1]), np.divide(meanTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanTLarge[2], yerr = np.minimum(meanTLargeRange[2], np.sqrt(meanTLarge[2]), np.divide(meanTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanTLarge[3], yerr = np.minimum(meanTLargeRange[3], np.sqrt(meanTLarge[3]), np.divide(meanTLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_large_mean.png")
plt.clf()

plt.errorbar(ldaset, minTLarge[0], yerr = np.minimum(minTLargeRange[0], np.sqrt(minTLarge[0]), np.divide(minTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTLarge[1], yerr = np.minimum(minTLargeRange[1], np.sqrt(minTLarge[1]), np.divide(minTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTLarge[2], yerr = np.minimum(minTLargeRange[2], np.sqrt(minTLarge[2]), np.divide(minTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTLarge[3], yerr = np.minimum(minTLargeRange[3], np.sqrt(minTLarge[3]), np.divide(minTLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_large_min.png")
plt.clf()

plt.errorbar(ldaset, maxTLarge[0], yerr = np.minimum(maxTLargeRange[0], np.sqrt(maxTLarge[0]), np.divide(maxTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTLarge[1], yerr = np.minimum(maxTLargeRange[1], np.sqrt(maxTLarge[1]), np.divide(maxTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTLarge[2], yerr = np.minimum(maxTLargeRange[2], np.sqrt(maxTLarge[2]), np.divide(maxTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTLarge[3], yerr = np.minimum(maxTLargeRange[3], np.sqrt(maxTLarge[3]), np.divide(maxTLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_large_max.png")
plt.clf()

# EXPERIMENT 4: % of noise vs ground truth for each T
plt.errorbar(Tset, meanPerc[0], yerr = np.minimum(meanPercRange[0], np.sqrt(meanPerc[0]), np.divide(meanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, meanPerc[1], yerr = np.minimum(meanPercRange[1], np.sqrt(meanPerc[1]), np.divide(meanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, meanPerc[2], yerr = np.minimum(meanPercRange[2], np.sqrt(meanPerc[2]), np.divide(meanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, meanPerc[3], yerr = np.minimum(meanPercRange[3], np.sqrt(meanPerc[3]), np.divide(meanPerc[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of T")
plt.ylabel("Noise (%)")
plt.savefig("Exp4_femnist_T_perc_mean.png")
plt.clf()

plt.errorbar(Tset, minPerc[0], yerr = np.minimum(minPercRange[0], np.sqrt(minPerc[0]), np.divide(minPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minPerc[1], yerr = np.minimum(minPercRange[1], np.sqrt(minPerc[1]), np.divide(minPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minPerc[2], yerr = np.minimum(minPercRange[2], np.sqrt(minPerc[2]), np.divide(minPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minPerc[3], yerr = np.minimum(minPercRange[3], np.sqrt(minPerc[3]), np.divide(minPerc[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of T")
plt.ylabel("Noise (%)")
plt.savefig("Exp4_femnist_T_perc_min.png")
plt.clf()

plt.errorbar(Tset, maxPerc[0], yerr = np.minimum(maxPercRange[0], np.sqrt(maxPerc[0]), np.divide(maxPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxPerc[1], yerr = np.minimum(maxPercRange[1], np.sqrt(maxPerc[1]), np.divide(maxPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxPerc[2], yerr = np.minimum(maxPercRange[2], np.sqrt(maxPerc[2]), np.divide(maxPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxPerc[3], yerr = np.minimum(maxPercRange[3], np.sqrt(maxPerc[3]), np.divide(maxPerc[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of T")
plt.ylabel("Noise (%)")
plt.savefig("Exp4_femnist_T_perc_max.png")
plt.clf()

# compute total runtime
totalTime = time.perf_counter() - startTime
hours = totalTime // 3600
minutes = (totalTime % 3600) // 60
seconds = totalTime % 60

if hours == 0:
    if minutes == 1:
        print(f"\nRuntime: {round(minutes)} minute {round(seconds, 2)} seconds.\n")
    else:
        print(f"\nRuntime: {round(minutes)} minutes {round(seconds, 2)} seconds.\n")
elif hours == 1:
    if minutes == 1:
        print(f"\nRuntime: {round(hours)} hour {round(minutes)} minute {round(seconds, 2)} seconds.\n")
    else:
        print(f"\nRuntime: {round(hours)} hour {round(minutes)} minutes {round(seconds, 2)} seconds.\n")
else:
    if minutes == 1:
        print(f"\nRuntime: {round(hours)} hours {round(minutes)} minute {round(seconds, 2)} seconds.\n")
    else:
        print(f"\nRuntime: {round(hours)} hours {round(minutes)} minutes {round(seconds, 2)} seconds.\n")