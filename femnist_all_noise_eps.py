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
T = int(numWriters / 20)
T1 = 11*T

# lists of the values of epsilon and lambda, as well as the trials that will be explored
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
trialset = ["Dist", "TAgg", "Trusted", "no_privacy"]
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

meanEstRange = np.zeros((TS, ES))
meanLdaOptRange = np.zeros((TS, ES))
meanEstZeroRange = np.zeros((TS, ES))
meanEstOneRange = np.zeros((TS, ES))
meanPercRange = np.zeros((TS, ES))
meanEpsSmallRange = np.zeros((TS, LS))
meanEpsDefRange = np.zeros((TS, LS))
meanEpsMidRange = np.zeros((TS, LS))
meanEpsLargeRange = np.zeros((TS, LS))

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

minEstRange = np.zeros((TS, ES))
minLdaOptRange = np.zeros((TS, ES))
minEstZeroRange = np.zeros((TS, ES))
minEstOneRange = np.zeros((TS, ES))
minPercRange = np.zeros((TS, ES))
minEpsSmallRange = np.zeros((TS, LS))
minEpsDefRange = np.zeros((TS, LS))
minEpsMidRange = np.zeros((TS, LS))
minEpsLargeRange = np.zeros((TS, LS))

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

maxEstRange = np.zeros((TS, ES))
maxLdaOptRange = np.zeros((TS, ES))
maxEstZeroRange = np.zeros((TS, ES))
maxEstOneRange = np.zeros((TS, ES))
maxPercRange = np.zeros((TS, ES))
maxEpsSmallRange = np.zeros((TS, LS))
maxEpsDefRange = np.zeros((TS, LS))
maxEpsMidRange = np.zeros((TS, LS))
maxEpsLargeRange = np.zeros((TS, LS))

# global parameters
ALPHA = 0.01 # smoothing parameter
E = 17 # size of subset for k3 estimator
DTA = 0.1
A = 0 # parameter for addition of noise
R1 = 90
ldaStep = 0.05
RS = 10
SEED_FREQ = 0
SMALL_INDEX = 0
DEF_INDEX = 3
MID_INDEX = 6
LARGE_INDEX = 10

for trial in range(4):
    meanfile = open(f"femnist_eps_{trialset[trial]}_mean.txt", "w", encoding = 'utf-8')
    minfile = open(f"femnist_eps_{trialset[trial]}_min.txt", "w", encoding = 'utf-8')
    maxfile = open(f"femnist_eps_{trialset[trial]}_max.txt", "w", encoding = 'utf-8')
    EPS_FREQ = 0

    for eps in epsset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        # temporary stores for each repeat
        tempMeanValue = np.zeros(RS)
        tempMeanEst = np.zeros(RS)
        tempMeanLdaOpt = np.zeros(RS)
        tempMeanEstZero = np.zeros(RS)
        tempMeanEstOne = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        tempMeanEpsSmall = np.zeros((LS, RS))
        tempMeanEpsDef = np.zeros((LS, RS))
        tempMeanEpsMid = np.zeros((LS, RS))
        tempMeanEpsLarge = np.zeros((LS, RS))
            
        tempMinValue = np.zeros(RS)
        tempMinEst = np.zeros(RS)
        tempMinLdaOpt = np.zeros(RS)
        tempMinEstZero = np.zeros(RS)
        tempMinEstOne = np.zeros(RS)
        tempMinPerc = np.zeros(RS)
        tempMinEpsSmall = np.zeros((LS, RS))
        tempMinEpsDef = np.zeros((LS, RS))
        tempMinEpsMid = np.zeros((LS, RS))
        tempMinEpsLarge = np.zeros((LS, RS))

        tempMaxValue = np.zeros(RS)
        tempMaxEst = np.zeros(RS)
        tempMaxLdaOpt = np.zeros(RS)
        tempMaxEstZero = np.zeros(RS)
        tempMaxEstOne = np.zeros(RS)
        tempMaxPerc = np.zeros(RS)
        tempMaxEpsSmall = np.zeros((LS, RS))
        tempMaxEpsDef = np.zeros((LS, RS))
        tempMaxEpsMid = np.zeros((LS, RS))
        tempMaxEpsLarge = np.zeros((LS, RS))

        for rep in range(RS):
            print(f"epsilon = {eps}, repeat {rep + 1}...")

            # initialising seeds for random sampling
            tf.random.set_seed(SEED_FREQ)
            np.random.seed(SEED_FREQ)

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
            b1 = (1 + log(2)) / eps
            b2 = (2*((log(1.25))/DTA)*b1) / eps
 
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
            tempMeanValue[rep] = np.mean(uList)
            tempMinValue[rep] = uList[minIndex]
            tempMaxValue[rep] = uList[maxIndex]
        
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

                # "TAgg" (intermediate server adds Gaussian noise term)
                if trial == 1:
                    meanLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    minLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))
                    maxLdaNoise[l] = gaussNoise.sample(sample_shape = (1,))

                    meanLda[l] = meanLda[l] + meanLdaNoise[l]
                    minLda[l] = minLda[l] + minLdaNoise[l]
                    maxLda[l] = maxLda[l] + maxLdaNoise[l]
            
                # mean / min / max across lambdas for eps = 0.05 (small)
                if EPS_FREQ == SMALL_INDEX:
                    tempMeanEpsSmall[l, rep] = meanLda[l]
                    tempMinEpsSmall[l, rep] = minLda[l]
                    tempMaxEpsSmall[l, rep] = maxLda[l]

                # eps = 0.5 (default)
                if EPS_FREQ == DEF_INDEX:
                    tempMeanEpsDef[l, rep] = meanLda[l]
                    tempMinEpsDef[l, rep] = minLda[l]
                    tempMaxEpsDef[l, rep] = maxLda[l]

                # eps = 1.5 (mid)
                if EPS_FREQ == MID_INDEX:
                    tempMeanEpsMid[l, rep] = meanLda[l]
                    tempMinEpsMid[l, rep] = minLda[l]
                    tempMaxEpsMid[l, rep] = maxLda[l]

                # eps = 3 (large)
                if EPS_FREQ == LARGE_INDEX:
                    tempMeanEpsLarge[l, rep] = meanLda[l]
                    tempMinEpsLarge[l, rep] = minLda[l]
                    tempMaxEpsLarge[l, rep] = maxLda[l]

            # find lambda that produces minimum error
            meanLdaIndex = np.argmin(meanLda)
            minLdaIndex = np.argmin(minLda)
            maxLdaIndex = np.argmin(maxLda)

            meanMinError = meanLda[meanLdaIndex]
            minMinError = minLda[minLdaIndex]
            maxMinError = maxLda[maxLdaIndex]

            # mean / min / max across clients for optimum lambda
            tempMeanEst[rep] = meanMinError
            tempMinEst[rep] = minMinError
            tempMaxEst[rep] = maxMinError

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
                lapNoise = tfp.distributions.Laplace(loc = A, scale = b1)
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
                tempMeanEst[rep] = (tempMeanEst[rep] + meanNoise - tempMeanValue[rep])**2
                tempMinEst[rep] = (tempMinEst[rep] + minNoise - tempMinValue[rep])**2
                tempMaxEst[rep] = (tempMaxEst[rep] + maxNoise - tempMaxValue[rep])**2

                # lambda = 0
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] + meanZeroNoise - tempMeanValue[rep])**2
                tempMinEstZero[rep] = (tempMinEstZero[rep] + minZeroNoise - tempMinValue[rep])**2
                tempMaxEstZero[rep] = (tempMaxEstZero[rep] + maxZeroNoise - tempMaxValue[rep])**2

                # lambda = 1
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] + meanOneNoise - tempMeanValue[rep])**2
                tempMinEstOne[rep] = (tempMinEstOne[rep] + minOneNoise - tempMinValue[rep])**2
                tempMaxEstOne[rep] = (tempMaxEstOne[rep] + maxOneNoise - tempMaxValue[rep])**2

                for l in range(LS):
        
                    # eps = 0.05 (small)
                    if EPS_FREQ == SMALL_INDEX:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        minSmallNoise = lapNoise.sample(sample_shape = (1,))
                        maxSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsSmall[l, rep] = (tempMeanEpsSmall[l, rep] + meanSmallNoise - tempMeanValue[rep])**2
                        tempMinEpsSmall[l, rep] = (tempMinEpsSmall[l, rep] + minSmallNoise - tempMinValue[rep])**2
                        tempMaxEpsSmall[l, rep] = (tempMaxEpsSmall[l, rep] + maxSmallNoise - tempMaxValue[rep])**2

                    # eps = 0.5 (def)
                    if EPS_FREQ == DEF_INDEX:
                        meanDefNoise = lapNoise.sample(sample_shape = (1,))
                        minDefNoise = lapNoise.sample(sample_shape = (1,))
                        maxDefNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsDef[l, rep] = (tempMeanEpsDef[l, rep] + meanDefNoise - tempMeanValue[rep])**2
                        tempMinEpsDef[l, rep] = (tempMinEpsDef[l, rep] + minDefNoise - tempMinValue[rep])**2
                        tempMaxEpsDef[l, rep] = (tempMaxEpsDef[l, rep] + maxDefNoise - tempMaxValue[rep])**2

                    # eps = 1.5 (mid)
                    if EPS_FREQ == MID_INDEX:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        minMidNoise = lapNoise.sample(sample_shape = (1,))
                        maxMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsMid[l, rep] = (tempMeanEpsMid[l, rep] + meanMidNoise - tempMeanValue[rep])**2
                        tempMinEpsMid[l, rep] = (tempMinEpsMid[l, rep] + minMidNoise - tempMinValue[rep])**2
                        tempMaxEpsMid[l, rep] = (tempMaxEpsMid[l, rep] + maxMidNoise - tempMaxValue[rep])**2

                    # eps = 3 (large)
                    if EPS_FREQ == LARGE_INDEX:
                        meanLargeNoise = lapNoise.sample(sample_shape = (1,))
                        minLargeNoise = lapNoise.sample(sample_shape = (1,))
                        maxLargeNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsLarge[l, rep] = (tempMeanEpsLarge[l, rep] + meanLargeNoise - tempMeanValue[rep])**2
                        tempMinEpsLarge[l, rep] = (tempMinEpsLarge[l, rep] + minLargeNoise - tempMinValue[rep])**2
                        tempMaxEpsLarge[l, rep] = (tempMaxEpsLarge[l, rep] + maxLargeNoise - tempMaxValue[rep])**2
        
            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEst[rep] = (tempMeanEst[rep] - tempMeanValue[rep])**2
                tempMinEst[rep] = (tempMinEst[rep] - tempMinValue[rep])**2
                tempMaxEst[rep] = (tempMaxEst[rep] - tempMaxValue[rep])**2

                # lambda = 0
                tempMeanEstZero[rep] = (tempMeanEstZero[rep] - tempMeanValue[rep])**2
                tempMinEstZero[rep] = (tempMinEstZero[rep] - tempMinValue[rep])**2
                tempMaxEstZero[rep] = (tempMaxEstZero[rep] - tempMaxValue[rep])**2

                # lambda = 1
                tempMeanEstOne[rep] = (tempMeanEstOne[rep] - tempMeanValue[rep])**2
                tempMinEstOne[rep] = (tempMinEstOne[rep] - tempMinValue[rep])**2
                tempMaxEstOne[rep] = (tempMaxEstOne[rep] - tempMaxValue[rep])**2

                for l in range(LS):
        
                    # eps = 0.05 (small)
                    if EPS_FREQ == SMALL_INDEX:
                        tempMeanEpsSmall[l, rep] = (tempMeanEpsSmall[l, rep] - tempMeanValue[rep])**2
                        tempMinEpsSmall[l, rep] = (tempMinEpsSmall[l, rep] - tempMinValue[rep])**2
                        tempMaxEpsSmall[l, rep] = (tempMaxEpsSmall[l, rep] - tempMaxValue[rep])**2

                    # eps = 0.5 (def)
                    if EPS_FREQ == DEF_INDEX:
                        tempMeanEpsDef[l, rep] = (tempMeanEpsDef[l, rep] - tempMeanValue[rep])**2
                        tempMinEpsDef[l, rep] = (tempMinEpsDef[l, rep] - tempMinValue[rep])**2
                        tempMaxEpsDef[l, rep] = (tempMaxEpsDef[l, rep] - tempMaxValue[rep])**2

                    # eps = 1.5 (mid)
                    if EPS_FREQ == MID_INDEX:
                        tempMeanEpsMid[l, rep] = (tempMeanEpsMid[l, rep] - tempMeanValue[rep])**2
                        tempMinEpsMid[l, rep] = (tempMinEpsMid[l, rep] - tempMinValue[rep])**2
                        tempMaxEpsMid[l, rep] = (tempMaxEpsMid[l, rep] - tempMaxValue[rep])**2

                    # eps = 3 (large)
                    if EPS_FREQ == LARGE_INDEX:
                        tempMeanEpsLarge[l, rep] = (tempMeanEpsLarge[l, rep] - tempMeanValue[rep])**2
                        tempMinEpsLarge[l, rep] = (tempMinEpsLarge[l, rep] - tempMinValue[rep])**2
                        tempMaxEpsLarge[l, rep] = (tempMaxEpsLarge[l, rep] - tempMaxValue[rep])**2
            
            # compute % of noise vs ground truth
            if trial == 0:
                tempMeanPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMeanValue[rep])))*100       
            if trial == 1:
                tempMeanPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMeanValue[rep]))*100
            if trial == 2:
                tempMeanPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMeanValue[rep])))*100

            if trial == 0:
                tempMinPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMinValue[rep])))*100
            if trial == 1:
                tempMinPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMinValue[rep]))*100
            if trial == 2:
                tempMinPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMinValue[rep])))*100
            
            if trial == 0:
                tempMaxPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMaxValue[rep])))*100
            if trial == 1:
                tempMaxPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMaxValue[rep]))*100
            if trial == 2:
                tempMaxPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMaxValue[rep])))*100
            
            SEED_FREQ = SEED_FREQ + 1

        # compute mean of repeats
        meanValue[trial, EPS_FREQ] = np.mean(tempMeanValue)
        meanEst[trial, EPS_FREQ] = np.mean(tempMeanEst)
        meanLdaOpt[trial, EPS_FREQ] = np.mean(tempMeanLdaOpt)
        meanEstZero[trial, EPS_FREQ] = np.mean(tempMeanEstZero)
        meanEstOne[trial, EPS_FREQ] = np.mean(tempMeanEstOne)
        meanPerc[trial, EPS_FREQ] = np.mean(tempMeanPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                meanEpsSmall[trial, l] = np.mean(tempMeanEpsSmall[l])
            if EPS_FREQ == DEF_INDEX:
                meanEpsDef[trial, l] = np.mean(tempMeanEpsDef[l])
            if EPS_FREQ == MID_INDEX:
                meanEpsMid[trial, l] = np.mean(tempMeanEpsMid[l])
            if EPS_FREQ == LARGE_INDEX:
                meanEpsLarge[trial, l] = np.mean(tempMeanEpsLarge[l])

        minValue[trial, EPS_FREQ] = np.mean(tempMinValue)
        minEst[trial, EPS_FREQ] = np.mean(tempMinEst)
        minLdaOpt[trial, EPS_FREQ] = np.mean(tempMinLdaOpt)
        minEstZero[trial, EPS_FREQ] = np.mean(tempMinEstZero)
        minEstOne[trial, EPS_FREQ] = np.mean(tempMeanEstOne)
        minPerc[trial, EPS_FREQ] = np.mean(tempMinPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                minEpsSmall[trial, l] = np.mean(tempMinEpsSmall[l])
            if EPS_FREQ == DEF_INDEX:
                minEpsDef[trial, l] = np.mean(tempMinEpsDef[l])
            if EPS_FREQ == MID_INDEX:
                minEpsMid[trial, l] = np.mean(tempMinEpsMid[l])
            if EPS_FREQ == LARGE_INDEX:
                minEpsLarge[trial, l] = np.mean(tempMinEpsLarge[l])

        maxValue[trial, EPS_FREQ] = np.mean(tempMaxValue)
        maxEst[trial, EPS_FREQ] = np.mean(tempMaxEst)
        maxLdaOpt[trial, EPS_FREQ] = np.mean(tempMaxLdaOpt)
        maxEstZero[trial, EPS_FREQ] = np.mean(tempMaxEstZero)
        maxEstOne[trial, EPS_FREQ] = np.mean(tempMaxEstOne)
        maxPerc[trial, EPS_FREQ] = np.mean(tempMaxPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                maxEpsSmall[trial, l] = np.mean(tempMaxEpsSmall[l])
            if EPS_FREQ == DEF_INDEX:
                maxEpsDef[trial, l] = np.mean(tempMaxEpsDef[l])
            if EPS_FREQ == MID_INDEX:
                maxEpsMid[trial, l] = np.mean(tempMaxEpsMid[l])
            if EPS_FREQ == LARGE_INDEX:
                maxEpsLarge[trial, l] = np.mean(tempMaxEpsLarge[l])

        # compute standard deviation of repeats
        meanEstRange[trial, EPS_FREQ] = np.std(tempMeanEst)
        meanLdaOptRange[trial, EPS_FREQ] = np.std(tempMeanLdaOpt)
        meanEstZeroRange[trial, EPS_FREQ] = np.std(tempMeanEstZero)
        meanEstOneRange[trial, EPS_FREQ] = np.std(tempMeanEstOne)
        meanPercRange[trial, EPS_FREQ] = np.std(tempMeanPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                meanEpsSmallRange[trial, l] = np.std(tempMeanEpsSmall[l])
            if EPS_FREQ == DEF_INDEX:
                meanEpsDefRange[trial, l] = np.std(tempMeanEpsDef[l])
            if EPS_FREQ == MID_INDEX:
                meanEpsMidRange[trial, l] = np.std(tempMeanEpsMid[l])
            if EPS_FREQ == LARGE_INDEX:
                meanEpsLargeRange[trial, l] = np.std(tempMeanEpsLarge[l])

        minEstRange[trial, EPS_FREQ] = np.std(tempMinEst)
        minLdaOptRange[trial, EPS_FREQ] = np.std(tempMinLdaOpt)
        minEstZeroRange[trial, EPS_FREQ] = np.std(tempMinEstZero)
        minEstOneRange[trial, EPS_FREQ] = np.std(tempMinEstOne)
        minPercRange[trial, EPS_FREQ] = np.std(tempMinPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                minEpsSmallRange[trial, l] = np.std(tempMinEpsSmall[l])
            if EPS_FREQ == DEF_INDEX:
                minEpsDefRange[trial, l] = np.std(tempMinEpsDef[l])
            if EPS_FREQ == MID_INDEX:
                minEpsMidRange[trial, l] = np.std(tempMinEpsMid[l])
            if EPS_FREQ == LARGE_INDEX:
                minEpsLargeRange[trial, l] = np.std(tempMinEpsLarge[l])

        maxEstRange[trial, EPS_FREQ] = np.std(tempMaxEst)
        maxLdaOptRange[trial, EPS_FREQ] = np.std(tempMaxLdaOpt)
        maxEstZeroRange[trial, EPS_FREQ] = np.std(tempMaxEstZero)
        maxEstOneRange[trial, EPS_FREQ] = np.std(tempMaxEstOne)
        maxPercRange[trial, EPS_FREQ] = np.std(tempMaxPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                maxEpsSmallRange[trial, l] = np.std(tempMaxEpsSmall[l])
            if EPS_FREQ == DEF_INDEX:
                maxEpsDefRange[trial, l] = np.std(tempMaxEpsDef[l])
            if EPS_FREQ == MID_INDEX:
                maxEpsMidRange[trial, l] = np.std(tempMaxEpsMid[l])
            if EPS_FREQ == LARGE_INDEX:
                maxEpsLargeRange[trial, l] = np.std(tempMaxEpsLarge[l])

        # write statistics on data files
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
        meanfile.write(f"Noise: {np.round(meanPerc[trial, EPS_FREQ], 2)}%\n")

        minfile.write(f"\nMin Pair: {minPair}\n")
        minfile.write(f"Min Error: {round(minEst[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Optimal Lambda: {round(minLdaOpt[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Ground Truth: {round(minValue[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Noise: {np.round(minPerc[trial, EPS_FREQ], 2)}%\n")

        maxfile.write(f"\nMax Pair: {maxPair}\n")
        maxfile.write(f"Max Error: {round(maxEst[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Optimal Lambda: {round(maxLdaOpt[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Ground Truth: {round(maxValue[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Noise: {np.round(maxPerc[trial, EPS_FREQ], 2)}%\n")

        EPS_FREQ = EPS_FREQ + 1

# plot error of PRIEST-KLD for each epsilon (mean)
plt.errorbar(epsset, meanEst[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEst[0]), np.divide(meanEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEst[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEst[1]), np.divide(meanEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEst[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEst[2]), np.divide(meanEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (min pair)
plt.errorbar(epsset, minEst[0], yerr = np.minimum(minEstRange[0], np.sqrt(minEst[0]), np.divide(minEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minEst[1], yerr = np.minimum(minEstRange[1], np.sqrt(minEst[1]), np.divide(minEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minEst[2], yerr = np.minimum(minEstRange[2], np.sqrt(minEst[2]), np.divide(minEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min.png")
plt.clf()

# plot error of PRIEST-KLD for each epsilon (max pair)
plt.errorbar(epsset, maxEst[0], yerr = np.minimum(maxEstRange[0], np.sqrt(maxEst[0]), np.divide(maxEst[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxEst[1], yerr = np.minimum(maxEstRange[1], np.sqrt(maxEst[1]), np.divide(maxEst[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxEst[2], yerr = np.minimum(maxEstRange[2], np.sqrt(maxEst[2]), np.divide(maxEst[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max.png")
plt.clf()

# plot optimum lambda for each epsilon (mean)
plt.errorbar(epsset, meanLdaOpt[0], yerr = np.minimum(meanLdaOptRange[0], np.sqrt(meanLdaOpt[0]), np.divide(meanLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanLdaOpt[1], yerr = np.minimum(meanLdaOptRange[1], np.sqrt(meanLdaOpt[1]), np.divide(meanLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanLdaOpt[2], yerr = np.minimum(meanLdaOptRange[2], np.sqrt(meanLdaOpt[2]), np.divide(meanLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_eps_lda_opt_mean.png")
plt.clf()

# plot optimum lambda for each epsilon (min pair)
plt.errorbar(epsset, minLdaOpt[0], yerr = np.minimum(minLdaOptRange[0], np.sqrt(minLdaOpt[0]), np.divide(minLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minLdaOpt[1], yerr = np.minimum(minLdaOptRange[1], np.sqrt(minLdaOpt[1]), np.divide(minLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minLdaOpt[2], yerr = np.minimum(minLdaOptRange[2], np.sqrt(minLdaOpt[2]), np.divide(minLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_eps_lda_opt_min.png")
plt.clf()

# plot optimum lambda for each epsilon (max pair)
plt.errorbar(epsset, maxLdaOpt[0], yerr = np.minimum(maxLdaOptRange[0], np.sqrt(meanLdaOpt[0]), np.divide(meanLdaOpt[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxLdaOpt[1], yerr = np.minimum(maxLdaOptRange[1], np.sqrt(meanLdaOpt[1]), np.divide(meanLdaOpt[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxLdaOpt[2], yerr = np.minimum(maxLdaOptRange[2], np.sqrt(meanLdaOpt[2]), np.divide(meanLdaOpt[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of PRIEST-KLD")
plt.savefig("Femnist_eps_lda_opt_max.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 0 for each epsilon (mean)
plt.errorbar(epsset, meanEstZero[0], yerr = np.minimum(meanEstZeroRange[0], np.sqrt(meanEstZero[0]), np.divide(meanEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstZero[1], yerr = np.minimum(meanEstZeroRange[1], np.sqrt(meanEstZero[1]), np.divide(meanEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstZero[2], yerr = np.minimum(meanEstZeroRange[2], np.sqrt(meanEstZero[2]), np.divide(meanEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (min pair)
plt.errorbar(epsset, minEstZero[0], yerr = np.minimum(minEstZeroRange[0], np.sqrt(minEstZero[0]), np.divide(minEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minEstZero[1], yerr = np.minimum(minEstZeroRange[1], np.sqrt(minEstZero[1]), np.divide(minEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minEstZero[2], yerr = np.minimum(minEstZeroRange[2], np.sqrt(minEstZero[2]), np.divide(minEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_lda_zero.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 0 for each epsilon (max pair)
plt.errorbar(epsset, maxEstZero[0], yerr = np.minimum(maxEstZeroRange[0], np.sqrt(maxEstZero[0]), np.divide(maxEstZero[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxEstZero[1], yerr = np.minimum(maxEstZeroRange[1], np.sqrt(maxEstZero[1]), np.divide(maxEstZero[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxEstZero[2], yerr = np.minimum(maxEstZeroRange[2], np.sqrt(maxEstZero[2]), np.divide(maxEstZero[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_lda_zero.png")
plt.clf()

# plot error of PRIEST-KLD when lambda = 1 for each epsilon (mean)
plt.errorbar(epsset, meanEstOne[0], yerr = np.minimum(meanEstOneRange[0], np.sqrt(meanEstOne[0]), np.divide(meanEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstOne[1], yerr = np.minimum(meanEstOneRange[1], np.sqrt(meanEstOne[1]), np.divide(meanEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstOne[2], yerr = np.minimum(meanEstOneRange[2], np.sqrt(meanEstOne[2]), np.divide(meanEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (min pair)
plt.errorbar(epsset, minEstOne[0], yerr = np.minimum(minEstOneRange[0], np.sqrt(minEstOne[0]), np.divide(minEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minEstOne[1], yerr = np.minimum(minEstOneRange[1], np.sqrt(minEstOne[1]), np.divide(minEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minEstOne[2], yerr = np.minimum(minEstOneRange[2], np.sqrt(minEstOne[2]), np.divide(minEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_lda_one.png")
plt.clf()

# plot error oF PRIEST-KLD when lambda = 1 for each epsilon (max pair)
plt.errorbar(epsset, maxEstOne[0], yerr = np.minimum(maxEstOneRange[0], np.sqrt(maxEstOne[0]), np.divide(maxEstOne[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxEstOne[1], yerr = np.minimum(maxEstOneRange[1], np.sqrt(maxEstOne[1]), np.divide(maxEstOne[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxEstOne[2], yerr = np.minimum(maxEstOneRange[2], np.sqrt(maxEstOne[2]), np.divide(maxEstOne[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_lda_one.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (mean)
plt.errorbar(ldaset, meanEpsSmall[0], yerr = np.minimum(meanEpsSmallRange[0], np.sqrt(meanEpsSmall[0]), np.divide(meanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsSmall[1], yerr = np.minimum(meanEpsSmallRange[1], np.sqrt(meanEpsSmall[1]), np.divide(meanEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsSmall[2], yerr = np.minimum(meanEpsSmallRange[2], np.sqrt(meanEpsSmall[2]), np.divide(meanEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsSmall[3], yerr = np.minimum(meanEpsSmallRange[3], np.sqrt(meanEpsSmall[3]), np.divide(meanEpsSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (min pair)
plt.errorbar(ldaset, minEpsSmall[0], yerr = np.minimum(minEpsSmallRange[0], np.sqrt(minEpsSmall[0]), np.divide(minEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsSmall[1], yerr = np.minimum(minEpsSmallRange[1], np.sqrt(minEpsSmall[1]), np.divide(minEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsSmall[2], yerr = np.minimum(minEpsSmallRange[2], np.sqrt(minEpsSmall[2]), np.divide(minEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsSmall[3], yerr = np.minimum(minEpsSmallRange[3], np.sqrt(minEpsSmall[3]), np.divide(minEpsSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.05 (max pair)
plt.errorbar(ldaset, maxEpsSmall[0], yerr = np.minimum(maxEpsSmallRange[0], np.sqrt(maxEpsSmall[0]), np.divide(maxEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsSmall[1], yerr = np.minimum(maxEpsSmallRange[1], np.sqrt(maxEpsSmall[2]), np.divide(maxEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsSmall[2], yerr = np.minimum(maxEpsSmallRange[2], np.sqrt(maxEpsSmall[2]), np.divide(maxEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsSmall[3], yerr = np.minimum(maxEpsSmallRange[3], np.sqrt(maxEpsSmall[3]), np.divide(maxEpsSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_small.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (mean)
plt.errorbar(ldaset, meanEpsDef[0], yerr = np.minimum(meanEpsDefRange[0], np.sqrt(meanEpsDef[0]), np.divide(meanEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsDef[1], yerr = np.minimum(meanEpsDefRange[1], np.sqrt(meanEpsDef[1]), np.divide(meanEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsDef[2], yerr = np.minimum(meanEpsDefRange[2], np.sqrt(meanEpsDef[2]), np.divide(meanEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsDef[3], yerr = np.minimum(meanEpsDefRange[3], np.sqrt(meanEpsDef[3]), np.divide(meanEpsDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (min pair)
plt.errorbar(ldaset, minEpsDef[0], yerr = np.minimum(minEpsDefRange[0], np.sqrt(minEpsDef[0]), np.divide(minEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsDef[1], yerr = np.minimum(minEpsDefRange[1], np.sqrt(minEpsDef[1]), np.divide(minEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsDef[2], yerr = np.minimum(minEpsDefRange[2], np.sqrt(minEpsDef[2]), np.divide(minEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsDef[3], yerr = np.minimum(minEpsDefRange[3], np.sqrt(minEpsDef[3]), np.divide(minEpsDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 0.5 (max pair)
plt.errorbar(ldaset, maxEpsDef[0], yerr = np.minimum(maxEpsDefRange[0], np.sqrt(maxEpsDef[0]), np.divide(maxEpsDef[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsDef[1], yerr = np.minimum(maxEpsDefRange[1], np.sqrt(maxEpsDef[1]), np.divide(maxEpsDef[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsDef[2], yerr = np.minimum(maxEpsDefRange[2], np.sqrt(maxEpsDef[2]), np.divide(maxEpsDef[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsDef[3], yerr = np.minimum(maxEpsDefRange[3], np.sqrt(maxEpsDef[3]), np.divide(maxEpsDef[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_def.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (mean)
plt.errorbar(ldaset, meanEpsMid[0], yerr = np.minimum(meanEpsMidRange[0], np.sqrt(meanEpsMid[0]), np.divide(meanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsMid[1], yerr = np.minimum(meanEpsMidRange[1], np.sqrt(meanEpsMid[1]), np.divide(meanEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsMid[2], yerr = np.minimum(meanEpsMidRange[2], np.sqrt(meanEpsMid[2]), np.divide(meanEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsMid[3], yerr = np.minimum(meanEpsMidRange[3], np.sqrt(meanEpsMid[3]), np.divide(meanEpsMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (min pair)
plt.errorbar(ldaset, minEpsMid[0], yerr = np.minimum(minEpsMidRange[0], np.sqrt(minEpsMid[0]), np.divide(minEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsMid[1], yerr = np.minimum(minEpsMidRange[1], np.sqrt(minEpsMid[1]), np.divide(minEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsMid[2], yerr = np.minimum(minEpsMidRange[2], np.sqrt(minEpsMid[2]), np.divide(minEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsMid[3], yerr = np.minimum(minEpsMidRange[3], np.sqrt(minEpsMid[3]), np.divide(minEpsMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 1.5 (max pair)
plt.errorbar(ldaset, maxEpsMid[0], yerr = np.minimum(maxEpsMidRange[0], np.sqrt(maxEpsMid[0]), np.divide(maxEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsMid[1], yerr = np.minimum(maxEpsMidRange[1], np.sqrt(maxEpsMid[1]), np.divide(maxEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsMid[2], yerr = np.minimum(maxEpsMidRange[2], np.sqrt(maxEpsMid[2]), np.divide(maxEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsMid[3], yerr = np.minimum(maxEpsMidRange[3], np.sqrt(maxEpsMid[3]), np.divide(maxEpsMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_mid.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (mean)
plt.errorbar(ldaset, meanEpsLarge[0], yerr = np.minimum(meanEpsLargeRange[0], np.sqrt(meanEpsLarge[0]), np.divide(meanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsLarge[1], yerr = np.minimum(meanEpsLargeRange[1], np.sqrt(meanEpsLarge[1]), np.divide(meanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsLarge[2], yerr = np.minimum(meanEpsLargeRange[2], np.sqrt(meanEpsLarge[2]), np.divide(meanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsLarge[3], yerr = np.minimum(meanEpsLargeRange[3], np.sqrt(meanEpsLarge[3]), np.divide(meanEpsLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_mean_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (min pair)
plt.errorbar(ldaset, minEpsLarge[0], yerr = np.minimum(minEpsLargeRange[0], np.sqrt(meanEpsLarge[0]), np.divide(meanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, minEpsLarge[1], yerr = np.minimum(minEpsLargeRange[1], np.sqrt(meanEpsLarge[1]), np.divide(meanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, minEpsLarge[2], yerr = np.minimum(minEpsLargeRange[2], np.sqrt(meanEpsLarge[2]), np.divide(meanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, minEpsLarge[3], yerr = np.minimum(minEpsLargeRange[3], np.sqrt(meanEpsLarge[3]), np.divide(meanEpsLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_min_eps_large.png")
plt.clf()

# plot error of PRIEST-KLD when epsilon = 3 (max pair)
plt.errorbar(ldaset, maxEpsLarge[0], yerr = np.minimum(maxEpsLargeRange[0], np.sqrt(meanEpsLarge[0]), np.divide(meanEpsLarge[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, maxEpsLarge[1], yerr = np.minimum(maxEpsLargeRange[1], np.sqrt(meanEpsLarge[1]), np.divide(meanEpsLarge[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, maxEpsLarge[2], yerr = np.minimum(maxEpsLargeRange[2], np.sqrt(meanEpsLarge[2]), np.divide(meanEpsLarge[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, maxEpsLarge[3], yerr = np.minimum(maxEpsLargeRange[3], np.sqrt(meanEpsLarge[3]), np.divide(meanEpsLarge[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of lambda")
plt.ylabel("Error of PRIEST-KLD")
plt.savefig("Femnist_eps_est_max_eps_large.png")
plt.clf()

# plot % of noise vs ground truth for each epsilon (mean)
plt.errorbar(epsset, meanPerc[0], yerr = np.minimum(meanPercRange[0], np.sqrt(meanPerc[0]), np.divide(meanPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanPerc[1], yerr = np.minimum(meanPercRange[1], np.sqrt(meanPerc[1]), np.divide(meanPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanPerc[2], yerr = np.minimum(meanPercRange[2], np.sqrt(meanPerc[2]), np.divide(meanPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
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
plt.errorbar(epsset, minPerc[0], yerr = np.minimum(minPercRange[0], np.sqrt(minPerc[0]), np.divide(minPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minPerc[1], yerr = np.minimum(minPercRange[1], np.sqrt(minPerc[1]), np.divide(minPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minPerc[2], yerr = np.minimum(minPercRange[2], np.sqrt(minPerc[2]), np.divide(minPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
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
plt.errorbar(epsset, maxPerc[0], yerr = np.minimum(maxPercRange[0], np.sqrt(maxPerc[0]), np.divide(maxPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxPerc[1], yerr = np.minimum(maxPercRange[1], np.sqrt(maxPerc[1]), np.divide(maxPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxPerc[2], yerr = np.minimum(maxPercRange[2], np.sqrt(maxPerc[2]), np.divide(maxPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([1, 10, 100, 500])
plt.ylim(0.1, 500)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of epsilon")
plt.ylabel("Noise (%)")
plt.savefig("Femnist_eps_perc_max.png")
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