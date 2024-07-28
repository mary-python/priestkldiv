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
meanPerc = np.zeros((TS, ES))

# related to min pairs
minValue = np.zeros((TS, ES))
minEstMSE = np.zeros((TS, ES))
minLdaOpt = np.zeros((TS, ES))
minDefDist = np.zeros((TS, ES))
minDefTAgg = np.zeros((TS, ES))
minLargeBest = np.zeros((TS, ES))
minPerc = np.zeros((TS, ES))
minTDef = np.zeros((TS, LS))
minTLarge = np.zeros((TS, LS))
minDefDistRange = np.zeros((TS, ES))
minDefTAggRange = np.zeros((TS, ES))
minLargeBestRange = np.zeros((TS, ES))
minTDefRange = np.zeros((TS, LS))
minTLargeRange = np.zeros((TS, LS))

# related to max pairs
maxValue = np.zeros((TS, ES))
maxEstMSE = np.zeros((TS, ES))
maxLdaOpt = np.zeros((TS, ES))
maxLargeDist = np.zeros((TS, ES))
maxLargeTAgg = np.zeros((TS, ES))
maxPerc = np.zeros((TS, ES))
maxTLarge = np.zeros((TS, LS))
maxLargeDistRange = np.zeros((TS, ES))
maxLargeTAggRange = np.zeros((TS, ES))
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
DEF_INDEX = 4
LARGE_INDEX = 10

for trial in range(4):
    meanfile = open(f"Data_{trialset[trial]}_T_a.txt", "w", encoding = 'utf-8')
    minfile = open(f"Data_{trialset[trial]}_T_b.txt", "w", encoding = 'utf-8')
    maxfile = open(f"Data_{trialset[trial]}_T_c.txt", "w", encoding = 'utf-8')
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
        tempMeanPerc = np.zeros(RS)
            
        tempMinValue = np.zeros(RS)
        tempMinInvValue = np.zeros(RS)
        tempMinRatio = np.zeros(RS)
        tempMinInvRatio = np.zeros(RS)
        tempMinEst = np.zeros(RS)
        tempMinInvEst = np.zeros(RS)
        tempMinEstMSE = np.zeros(RS)
        tempMinLdaOpt = np.zeros(RS)
        tempMinDefDist = np.zeros(RS)
        tempMinDefTAgg = np.zeros(RS)
        tempMinLargeBest = np.zeros(RS)
        tempMinPerc = np.zeros(RS)
        tempMinTDef = np.zeros((LS, RS))
        tempMinTLarge = np.zeros((LS, RS))

        tempMaxValue = np.zeros(RS)
        tempMaxInvValue = np.zeros(RS)
        tempMaxRatio = np.zeros(RS)
        tempMaxInvRatio = np.zeros(RS)
        tempMaxEst = np.zeros(RS)
        tempMaxInvEst = np.zeros(RS)
        tempMaxEstMSE = np.zeros(RS)
        tempMaxLdaOpt = np.zeros(RS)
        tempMaxLargeDist = np.zeros(RS)
        tempMaxLargeTAgg = np.zeros(RS)
        tempMaxPerc = np.zeros(RS)
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
            
                # min across lambdas for T = 180 (~5% of clients, default)
                if T_FREQ == DEF_INDEX:
                    tempMinTDef[l, rep] = minLda[l]

                # min / max across lambdas for T = 600 (~16.6% of clients, large)
                if T_FREQ == LARGE_INDEX:
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

            # T = 180, lambda = 0.45
            tempMinDefDist[rep] = minLda[9]

            # lambda = 0.2
            tempMinDefTAgg[rep] = minLda[4]

            # T = 540, lambda = 0.15
            tempMinLargeBest[rep] = maxLda[3]

            # lambda = 0.2
            tempMaxLargeDist[rep] = maxLda[4]

            # lambda = 0.3
            tempMaxLargeTAgg[rep] = maxLda[6]

            # "Trusted" (server adds Laplace noise term to final result)
            if trial == 2:
                lapNoise = tfp.distributions.Normal(loc = A, scale = b2)
                meanNoise = lapNoise.sample(sample_shape = (1,))
                minNoise = lapNoise.sample(sample_shape = (1,))
                maxNoise = lapNoise.sample(sample_shape = (1,))
                minDefDistNoise = lapNoise.sample(sample_shape = (1,))
                minDefTAggNoise = lapNoise.sample(sample_shape = (1,))
                maxLargeDistNoise = lapNoise.sample(sample_shape = (1,))
                maxLargeTAggNoise = lapNoise.sample(sample_shape = (1,))
                minLargeBestNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEstMSE[rep] = (tempMeanEst[rep] + meanNoise - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] + minNoise - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] + maxNoise - tempMaxValue[rep])**2

                tempMinDefDist[rep] = (tempMinDefDist[rep] + minDefDistNoise - tempMinValue[rep])**2
                tempMinDefTAgg[rep] = (tempMinDefTAgg[rep] + minDefTAggNoise - tempMinValue[rep])**2
                tempMinLargeBest[rep] = (tempMinLargeBest[rep] + minLargeBestNoise - tempMinValue[rep])**2
                tempMaxLargeDist[rep] = (tempMaxLargeDist[rep] + maxLargeDistNoise - tempMaxValue[rep])**2
                tempMaxLargeTAgg[rep] = (tempMaxLargeTAgg[rep] + maxLargeTAggNoise - tempMaxValue[rep])**2

                for l in range(LS):

                    # T = 180 (def)
                    if T_FREQ == DEF_INDEX:
                        minDefNoise = lapNoise.sample(sample_shape = (1,))
                        tempMinTDef[l, rep] = (tempMinTDef[l, rep] + minDefNoise - tempMinValue[rep])**2

                    # T = 540 (large)
                    if T_FREQ == LARGE_INDEX:
                        minLargeNoise = lapNoise.sample(sample_shape = (1,))
                        maxLargeNoise = lapNoise.sample(sample_shape = (1,))
                        tempMinTLarge[l, rep] = (tempMinTLarge[l, rep] + minLargeNoise - tempMinValue[rep])**2
                        tempMaxTLarge[l, rep] = (tempMaxTLarge[l, rep] + maxLargeNoise - tempMaxValue[rep])**2
        
            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEstMSE[rep] = (tempMeanEst[rep] - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] - tempMaxValue[rep])**2

                tempMinDefDist[rep] = (tempMinDefDist[rep] - tempMinValue[rep])**2
                tempMinDefTAgg[rep] = (tempMinDefTAgg[rep] - tempMinValue[rep])**2
                tempMinLargeBest[rep] = (tempMinLargeBest[rep] - tempMinValue[rep])**2
                tempMaxLargeDist[rep] = (tempMaxLargeDist[rep] - tempMaxValue[rep])**2
                tempMaxLargeTAgg[rep] = (tempMaxLargeTAgg[rep] - tempMaxValue[rep])**2

                for l in range(LS):

                    # T = 180 (def)
                    if T_FREQ == DEF_INDEX:
                        tempMinTDef[l, rep] = (tempMinTDef[l, rep] - tempMinValue[rep])**2

                    # T = 540 (large)
                    if T_FREQ == LARGE_INDEX:
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
        meanPerc[trial, T_FREQ] = np.mean(tempMeanPerc)
        meanEst = np.mean(tempMeanEst)
        meanInvEst = np.mean(tempMeanInvEst)

        minValue[trial, T_FREQ] = np.mean(tempMinValue)
        minEstMSE[trial, T_FREQ] = np.mean(tempMinEstMSE)
        minLdaOpt[trial, T_FREQ] = np.mean(tempMinLdaOpt)
        minDefDist[trial, T_FREQ] = np.mean(tempMinDefDist)
        minDefTAgg[trial, T_FREQ] = np.mean(tempMinDefTAgg)
        minLargeBest[trial, T_FREQ] = np.mean(tempMinLargeBest)
        minPerc[trial, T_FREQ] = np.mean(tempMinPerc)
        minEst = np.mean(tempMinEst)
        minInvEst = np.mean(tempMinInvEst)

        for l in range(LS):
            if T_FREQ == DEF_INDEX:
                minTDef[trial, l] = np.mean(tempMinTDef[l])
            if T_FREQ == LARGE_INDEX:
                minTLarge[trial, l] = np.mean(tempMinTLarge[l])

        maxValue[trial, T_FREQ] = np.mean(tempMaxValue)
        maxEstMSE[trial, T_FREQ] = np.mean(tempMaxEstMSE)
        maxLdaOpt[trial, T_FREQ] = np.mean(tempMaxLdaOpt)
        maxLargeDist[trial, T_FREQ] = np.mean(tempMaxLargeDist)
        maxLargeTAgg[trial, T_FREQ] = np.mean(tempMaxLargeTAgg)
        maxPerc[trial, T_FREQ] = np.mean(tempMaxPerc)
        maxEst = np.mean(tempMaxEst)
        maxInvEst = np.mean(tempMaxInvEst)

        for l in range(LS):
            if T_FREQ == LARGE_INDEX:
                maxTLarge[trial, l] = np.mean(tempMaxTLarge[l])

        minDefDistRange[trial, T_FREQ] = np.std(tempMinDefDist)
        minDefTAggRange[trial, T_FREQ] = np.std(tempMinDefTAgg)
        minLargeBestRange[trial, T_FREQ] = np.std(tempMinLargeBest)

        for l in range(LS):
            if T_FREQ == DEF_INDEX:
                minTDefRange[trial, l] = np.std(tempMinTDef[l])
            if T_FREQ == LARGE_INDEX:
                minTLargeRange[trial, l] = np.std(tempMinTLarge[l])
   
        maxLargeDistRange[trial, T_FREQ] = np.std(tempMaxLargeDist)
        maxLargeTAggRange[trial, T_FREQ] = np.std(tempMaxLargeTAgg)

        for l in range(LS):
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

# EXPERIMENT 2: MSE of PRIEST-KLD for fixed T (180, 540)
plt.errorbar(ldaset, minTDef[0], yerr = np.minimum(minTDefRange[0], np.sqrt(minTDef[0]), np.divide(minTDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTDef[1], yerr = np.minimum(minTDefRange[1], np.sqrt(minTDef[1]), np.divide(minTDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTDef[2], yerr = np.minimum(minTDefRange[2], np.sqrt(minTDef[2]), np.divide(minTDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTDef[3], yerr = np.minimum(minTDefRange[3], np.sqrt(minTDef[3]), np.divide(minTDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.01, 4000)
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_est_180.png")
plt.clf()

plt.errorbar(ldaset, minTLarge[0], yerr = np.minimum(minTLargeRange[0], np.sqrt(minTLarge[0]), np.divide(minTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minTLarge[1], yerr = np.minimum(minTLargeRange[1], np.sqrt(minTLarge[1]), np.divide(minTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minTLarge[2], yerr = np.minimum(minTLargeRange[2], np.sqrt(minTLarge[2]), np.divide(minTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minTLarge[3], yerr = np.minimum(minTLargeRange[3], np.sqrt(minTLarge[3]), np.divide(minTLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.ylim(0.05, 3000)
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_est_540_a.png")
plt.clf()

plt.errorbar(ldaset, maxTLarge[0], yerr = np.minimum(maxTLargeRange[0], np.sqrt(maxTLarge[0]), np.divide(maxTLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxTLarge[1], yerr = np.minimum(maxTLargeRange[1], np.sqrt(maxTLarge[1]), np.divide(maxTLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxTLarge[2], yerr = np.minimum(maxTLargeRange[2], np.sqrt(maxTLarge[2]), np.divide(maxTLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxTLarge[3], yerr = np.minimum(maxTLargeRange[3], np.sqrt(maxTLarge[3]), np.divide(maxTLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp2_femnist_T_est_540_b.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD for best lambdas extracted from experiment 2
plt.errorbar(Tset, minDefDist[0], yerr = np.minimum(minDefDistRange[0], np.sqrt(minDefDist[0]), np.divide(minDefDist[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minDefDist[1], yerr = np.minimum(minDefDistRange[1], np.sqrt(minDefDist[1]), np.divide(minDefDist[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minDefDist[2], yerr = np.minimum(minDefDistRange[2], np.sqrt(minDefDist[2]), np.divide(minDefDist[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minDefDist[3], yerr = np.minimum(minDefDistRange[3], np.sqrt(minDefDist[3]), np.divide(minDefDist[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.01, 3000)
plt.xlabel("Number of clients n")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_best_180_a.png")
plt.clf()

plt.errorbar(Tset, minDefTAgg[0], yerr = np.minimum(minDefTAggRange[0], np.sqrt(minDefTAgg[0]), np.divide(minDefTAgg[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minDefTAgg[1], yerr = np.minimum(minDefTAggRange[1], np.sqrt(minDefTAgg[1]), np.divide(minDefTAgg[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minDefTAgg[2], yerr = np.minimum(minDefTAggRange[2], np.sqrt(minDefTAgg[2]), np.divide(minDefTAgg[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minDefTAgg[3], yerr = np.minimum(minDefTAggRange[3], np.sqrt(minDefTAgg[3]), np.divide(minDefTAgg[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.01, 3000)
plt.xlabel("Number of clients n")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_best_180_b.png")
plt.clf()

plt.errorbar(Tset, minLargeBest[0], yerr = np.minimum(minLargeBestRange[0], np.sqrt(minLargeBest[0]), np.divide(minLargeBest[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, minLargeBest[1], yerr = np.minimum(minLargeBestRange[1], np.sqrt(minLargeBest[1]), np.divide(minLargeBest[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, minLargeBest[2], yerr = np.minimum(minLargeBestRange[2], np.sqrt(minLargeBest[2]), np.divide(minLargeBest[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, minLargeBest[3], yerr = np.minimum(minLargeBestRange[3], np.sqrt(minLargeBest[3]), np.divide(minLargeBest[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.01, 4000)
plt.xlabel("Number of clients n")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_best_540_a.png")
plt.clf()

plt.errorbar(Tset, maxLargeDist[0], yerr = np.minimum(maxLargeDistRange[0], np.sqrt(maxLargeDist[0]), np.divide(maxLargeDist[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxLargeDist[1], yerr = np.minimum(maxLargeDistRange[1], np.sqrt(maxLargeDist[1]), np.divide(maxLargeDist[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxLargeDist[2], yerr = np.minimum(maxLargeDistRange[2], np.sqrt(maxLargeDist[2]), np.divide(maxLargeDist[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxLargeDist[3], yerr = np.minimum(maxLargeDistRange[3], np.sqrt(maxLargeDist[3]), np.divide(maxLargeDist[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(1, 4000)
plt.xlabel("Number of clients n")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_best_540_b.png")
plt.clf()

plt.errorbar(Tset, maxLargeTAgg[0], yerr = np.minimum(maxLargeTAggRange[0], np.sqrt(maxLargeTAgg[0]), np.divide(maxLargeTAgg[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(Tset, maxLargeTAgg[1], yerr = np.minimum(maxLargeTAggRange[1], np.sqrt(maxLargeTAgg[1]), np.divide(maxLargeTAgg[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(Tset, maxLargeTAgg[2], yerr = np.minimum(maxLargeTAggRange[2], np.sqrt(maxLargeTAgg[2]), np.divide(maxLargeTAgg[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(Tset, maxLargeTAgg[3], yerr = np.minimum(maxLargeTAggRange[3], np.sqrt(maxLargeTAgg[3]), np.divide(maxLargeTAgg[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(1, 2000)
plt.xlabel("Number of clients n")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_T_best_540_c.png")
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