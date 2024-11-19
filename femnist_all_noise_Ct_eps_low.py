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
plt.rcParams.update({'font.size': 12})

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
Ctset = [36, 72, 108, 144, 180, 225, 270, 315, 360, 420, 480, 540, 600, 660, 720]
ES = len(Ctset)

# list of the lambdas and trials that will be explored
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
          1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
trialset = ["Dist", "TAgg", "Trusted", "no_privacy"]
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanValue = np.zeros((TS, ES))
meanEstMSE = np.zeros((TS, ES))
meanTSmall = np.zeros((TS, LS))
meanTMid = np.zeros((TS, LS))
meanTLarge = np.zeros((TS, LS))

meanEstRange = np.zeros((TS, ES))
meanTSmallRange = np.zeros((TS, LS))
meanTMidRange = np.zeros((TS, LS))
meanTLargeRange = np.zeros((TS, LS))

# related to min pairs
minValue = np.zeros((TS, ES))
minEstMSE = np.zeros((TS, ES))
minTSmall = np.zeros((TS, LS))
minTMid = np.zeros((TS, LS))
minTLarge = np.zeros((TS, LS))

minEstRange = np.zeros((TS, ES))
minTSmallRange = np.zeros((TS, LS))
minTMidRange = np.zeros((TS, LS))
minTLargeRange = np.zeros((TS, LS))

# related to max pairs
maxValue = np.zeros((TS, ES))
maxEstMSE = np.zeros((TS, ES))
maxTSmall = np.zeros((TS, LS))
maxTMid = np.zeros((TS, LS))
maxTLarge = np.zeros((TS, LS))

maxEstRange = np.zeros((TS, ES))
maxTSmallRange = np.zeros((TS, LS))
maxTMidRange = np.zeros((TS, LS))
maxTLargeRange = np.zeros((TS, LS))

# global parameters
ALPHA = 0.01 # smoothing parameter
E = 17 # size of subset for k3 estimator
EPS = 0.5
DTA = 0.05
A = 0 # parameter for addition of noise
R = 90
T = 10
SEED_FREQ = 0
SMALL_INDEX = 0
MID_INDEX = 6
LARGE_INDEX = 14

for trial in range(4):

    Ct_FREQ = 0

    for Ct in Ctset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        # temporary stores for each round
        tempMeanValue = np.zeros(T)
        tempMeanEst = np.zeros(T)
        tempMeanEstMSE = np.zeros(T)
        tempMeanTSmall = np.zeros((LS, T))
        tempMeanTMid = np.zeros((LS, T))
        tempMeanTLarge = np.zeros((LS, T))
            
        tempMinValue = np.zeros(T)
        tempMinEst = np.zeros(T)
        tempMinEstMSE = np.zeros(T)
        tempMinTSmall = np.zeros((LS, T))
        tempMinTMid = np.zeros((LS, T))
        tempMinTLarge = np.zeros((LS, T))

        tempMaxValue = np.zeros(T)
        tempMaxEst = np.zeros(T)
        tempMaxEstMSE = np.zeros(T)
        tempMaxTSmall = np.zeros((LS, T))
        tempMaxTMid = np.zeros((LS, T))
        tempMaxTLarge = np.zeros((LS, T))

        for rep in range(T):      
            print(f"Ct = {Ct}, round {rep + 1}...")

            # initialising seeds for random sampling
            tf.random.set_seed(SEED_FREQ)
            np.random.seed(SEED_FREQ)

            # store Ct images corresponding to each digit
            sampledWriters = np.random.choice(numWriters, Ct, replace = False)
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
            Ct1 = 11*Ct

            def smoothed_prob(dset, dig, im, ufreq):
                """Method to compute frequencies of unique images and return smoothed probabilities."""
                where = np.where(np.all(im == dset, axis = (1, 2)))
                freq = len(where[0])
                uImageSet[dig, ufreq] = im
                uFreqSet[dig, ufreq] = int(freq)
                uProbsSet[dig, ufreq] = float((freq + ALPHA)/(Ct1 + (ALPHA*(digFreq[dig]))))

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
            Ct2 = (11/3)*Ct*(E/U) # change this term so probabilities add up to 1

            # borrow data from corresponding indices of main image and frequency sets
            for D in range(0, 10):
                for i in range(E):
                    eImageSet[D, i] = uImageSet[D, uSampledSet[i]]
                    eFreqSet[D, i] = uFreqSet[D, uSampledSet[i]]
                    eTotalFreq[D] = sum(eFreqSet[D])
                    eProbsSet[D, i] = float((eFreqSet[D, i] + ALPHA)/(Ct2 + (ALPHA*(eTotalFreq[D]))))

            # load Gaussian noise distributions for clients and intermediate server
            b1 = (1 + log(2)) / EPS
            b2 = (2*((log(1.25))/DTA)*b1) / EPS

            if trial < 2:
                s = b2 * (np.sqrt(2) / R)
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

                        # "Dist" (each client adds Gaussian noise term)
                        if trial == 0:
                            startSample = abs(probGaussNoise.sample(sample_shape = (1,)))
                            startNoise.append(startSample)
                            nDist[C, D, j] = nDist[C, D, j] + startSample

                    # compute ratio between exact unknown distributions
                    ratio = abs(sum(nDist[C, D]) / sum(uDist[C, D]))

                    # eliminate all divide by zero errors
                    if ratio != 0.0 and sum(uDist[C, D]) != 0.0:
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
                    uRangeEst = abs(lda * (np.exp(uLogr) - 1) - uLogr)

                    # share PRIEST-KLD with intermediate server
                    uEst[LDA_FREQ, R_FREQ] = uRangeEst
                    LDA_FREQ = LDA_FREQ + 1

                R_FREQ = R_FREQ + 1
        
            # extract position of min and max pairs
            minIndex = np.argmin(uList)
            maxIndex = np.argmax(uList)

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
            for l in range(LS):
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
            
                # mean / min / max across lambdas for Ct = 36 (~1% of clients, small)
                if Ct_FREQ == SMALL_INDEX:
                    tempMeanTSmall[l, rep] = meanLda[l]
                    tempMinTSmall[l, rep] = minLda[l]
                    tempMaxTSmall[l, rep] = maxLda[l]

                # Ct = 270 (~7.5% of clients, mid)
                if Ct_FREQ == MID_INDEX:
                    tempMeanTMid[l, rep] = meanLda[l]
                    tempMinTMid[l, rep] = minLda[l]
                    tempMaxTMid[l, rep] = maxLda[l]

                # Ct = 720 (~20% of clients, large)
                if Ct_FREQ == LARGE_INDEX:
                    tempMeanTLarge[l, rep] = meanLda[l]
                    tempMinTLarge[l, rep] = minLda[l]
                    tempMaxTLarge[l, rep] = maxLda[l]

            # choose best lambda from experiment 1
            meanLdaIndex = 1
            minLdaIndex = 5
            maxLdaIndex = 1

            # mean / min / max across clients for best lambda
            tempMeanEst[rep] = meanLda[meanLdaIndex]
            tempMinEst[rep] = minLda[minLdaIndex]
            tempMaxEst[rep] = maxLda[maxLdaIndex]

            # "Trusted" (server adds Laplace noise term to final result)
            if trial == 2:
                lapNoise = tfp.distributions.Normal(loc = A, scale = b2)
                meanNoise = lapNoise.sample(sample_shape = (1,))
                minNoise = lapNoise.sample(sample_shape = (1,))
                maxNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEstMSE[rep] = (tempMeanEst[rep] + meanNoise - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] + minNoise - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] + maxNoise - tempMaxValue[rep])**2

                for l in range(LS):

                    # Ct = 36 (small)
                    if Ct_FREQ == SMALL_INDEX:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        minSmallNoise = lapNoise.sample(sample_shape = (1,))
                        maxSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanTSmall[l, rep] = (tempMeanTSmall[l, rep] + meanSmallNoise - tempMeanValue[rep])**2
                        tempMinTSmall[l, rep] = (tempMinTSmall[l, rep] + minSmallNoise - tempMinValue[rep])**2
                        tempMaxTSmall[l, rep] = (tempMaxTSmall[l, rep] + maxSmallNoise - tempMaxValue[rep])**2

                    # Ct = 270 (mid)
                    if Ct_FREQ == MID_INDEX:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        minMidNoise = lapNoise.sample(sample_shape = (1,))
                        maxMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanTMid[l, rep] = (tempMeanTMid[l, rep] + meanMidNoise - tempMeanValue[rep])**2
                        tempMinTMid[l, rep] = (tempMinTMid[l, rep] + minMidNoise - tempMinValue[rep])**2
                        tempMaxTMid[l, rep] = (tempMaxTMid[l, rep] + maxMidNoise - tempMaxValue[rep])**2

                    # Ct = 720 (large)
                    if Ct_FREQ == LARGE_INDEX:
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

                for l in range(LS):
                    
                    # Ct = 36 (small)
                    if Ct_FREQ == SMALL_INDEX:
                        tempMeanTSmall[l, rep] = (tempMeanTSmall[l, rep] - tempMeanValue[rep])**2
                        tempMinTSmall[l, rep] = (tempMinTSmall[l, rep] - tempMinValue[rep])**2
                        tempMaxTSmall[l, rep] = (tempMaxTSmall[l, rep] - tempMaxValue[rep])**2

                    # Ct = 270 (mid)
                    if Ct_FREQ == MID_INDEX:
                        tempMeanTMid[l, rep] = (tempMeanTMid[l, rep] - tempMeanValue[rep])**2
                        tempMinTMid[l, rep] = (tempMinTMid[l, rep] - tempMinValue[rep])**2
                        tempMaxTMid[l, rep] = (tempMaxTMid[l, rep] - tempMaxValue[rep])**2

                    # Ct = 720 (large)
                    if Ct_FREQ == LARGE_INDEX:
                        tempMeanTLarge[l, rep] = (tempMeanTLarge[l, rep] - tempMeanValue[rep])**2
                        tempMinTLarge[l, rep] = (tempMinTLarge[l, rep] - tempMinValue[rep])**2
                        tempMaxTLarge[l, rep] = (tempMaxTLarge[l, rep] - tempMaxValue[rep])**2
        
            SEED_FREQ = SEED_FREQ + 1

        # compute mean of rounds
        meanValue[trial, Ct_FREQ] = np.mean(tempMeanValue)
        meanEstMSE[trial, Ct_FREQ] = np.mean(tempMeanEstMSE)

        minValue[trial, Ct_FREQ] = np.mean(tempMinValue)
        minEstMSE[trial, Ct_FREQ] = np.mean(tempMinEstMSE)

        maxValue[trial, Ct_FREQ] = np.mean(tempMaxValue)
        maxEstMSE[trial, Ct_FREQ] = np.mean(tempMaxEstMSE)

        for l in range(LS):
            if Ct_FREQ == SMALL_INDEX:
                meanTSmall[trial, l] = np.mean(tempMeanTSmall[l])
                minTSmall[trial, l] = np.mean(tempMinTSmall[l])
                maxTSmall[trial, l] = np.mean(tempMaxTSmall[l])
            if Ct_FREQ == MID_INDEX:
                meanTMid[trial, l] = np.mean(tempMeanTMid[l])
                minTMid[trial, l] = np.mean(tempMinTMid[l])
                maxTMid[trial, l] = np.mean(tempMaxTMid[l])
            if Ct_FREQ == LARGE_INDEX:
                meanTLarge[trial, l] = np.mean(tempMeanTLarge[l])
                minTLarge[trial, l] = np.mean(tempMinTLarge[l])
                maxTLarge[trial, l] = np.mean(tempMaxTLarge[l])
        
        meanEstRange[trial, Ct_FREQ] = np.std(tempMeanEstMSE)
        minEstRange[trial, Ct_FREQ] = np.std(tempMinEstMSE)
        maxEstRange[trial, Ct_FREQ] = np.std(tempMaxEstMSE)

        for l in range(LS):
            if Ct_FREQ == SMALL_INDEX:
                meanTSmallRange[trial, l] = np.std(tempMeanTSmall[l])
                minTSmallRange[trial, l] = np.std(tempMinTSmall[l])
                maxTSmallRange[trial, l] = np.std(tempMaxTSmall[l])
            if Ct_FREQ == MID_INDEX:
                meanTMidRange[trial, l] = np.std(tempMeanTMid[l])
                minTMidRange[trial, l] = np.std(tempMinTMid[l])
                maxTMidRange[trial, l] = np.std(tempMaxTMid[l])
            if Ct_FREQ == LARGE_INDEX:
                meanTLargeRange[trial, l] = np.std(tempMeanTLarge[l])
                minTLargeRange[trial, l] = np.std(tempMinTLarge[l])
                maxTLargeRange[trial, l] = np.std(tempMaxTLarge[l])

        Ct_FREQ = Ct_FREQ + 1

upldaset = np.zeros(LS, dtype = bool)
loldaset = np.ones(LS, dtype = bool)
upTset = np.zeros(ES, dtype = bool)
loTset = np.ones(ES, dtype = bool)

# EXPERIMENT 1: MSE of PRIEST-KLD for fixed Ct (36, 270, 720)
fig, ax1 = plt.subplots(layout = 'constrained')
plotline1a, caplines1a, barlinecols1a = ax1.errorbar(ldaset, meanTSmall[0], yerr = np.minimum(meanTSmallRange[0], np.sqrt(meanTSmall[0]), np.divide(meanTSmall[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline1b, caplines1b, barlinecols1b = ax1.errorbar(ldaset, meanTSmall[1], yerr = np.minimum(meanTSmallRange[1], np.sqrt(meanTSmall[1]), np.divide(meanTSmall[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline1c, caplines1c, barlinecols1c = ax1.errorbar(ldaset, meanTSmall[2], yerr = np.minimum(meanTSmallRange[2], np.sqrt(meanTSmall[2]), np.divide(meanTSmall[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline1d, caplines1d, barlinecols1d = ax1.errorbar(ldaset, meanTSmall[3], yerr = np.minimum(meanTSmallRange[3], np.sqrt(meanTSmall[3]), np.divide(meanTSmall[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines1a[0].set_marker('')
caplines1b[0].set_marker('')
caplines1c[0].set_marker('')
caplines1d[0].set_marker('')
handles1, labels1 = ax1.get_legend_handles_labels()
handles1 = [h1[0] for h1 in handles1]
ax1.legend(handles1, labels1, loc = 'best')
ax1.set_yscale('log')
ax1.set_ylim(0.03, 20000)
ax1.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax1.set_ylabel("MSE of PRIEST-KLD")
ax1.figure.savefig("Eps_low_exp1_femnist_Ct_est_a_36.png")
plt.close()

fig, ax2 = plt.subplots(layout = 'constrained')
plotline2a, caplines2a, barlinecols2a = ax2.errorbar(ldaset, meanTMid[0], yerr = np.minimum(meanTMidRange[0], np.sqrt(meanTMid[0]), np.divide(meanTMid[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline2b, caplines2b, barlinecols2b = ax2.errorbar(ldaset, meanTMid[1], yerr = np.minimum(meanTMidRange[1], np.sqrt(meanTMid[1]), np.divide(meanTMid[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline2c, caplines2c, barlinecols2c = ax2.errorbar(ldaset, meanTMid[2], yerr = np.minimum(meanTMidRange[2], np.sqrt(meanTMid[2]), np.divide(meanTMid[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline2d, caplines2d, barlinecols2d = ax2.errorbar(ldaset, meanTMid[3], yerr = np.minimum(meanTMidRange[3], np.sqrt(meanTMid[3]), np.divide(meanTMid[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines2a[0].set_marker('')
caplines2b[0].set_marker('')
caplines2c[0].set_marker('')
caplines2d[0].set_marker('')
handles2, labels2 = ax2.get_legend_handles_labels()
handles2 = [h2[0] for h2 in handles2]
ax2.legend(handles2, labels2, loc = 'lower right')
ax2.set_yscale('log')
ax2.set_ylim(0.01, 20000)
ax2.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax2.set_ylabel("MSE of PRIEST-KLD")
ax2.figure.savefig("Eps_low_exp1_femnist_Ct_est_a_270.png")
plt.close()

fig, ax3 = plt.subplots(layout = 'constrained')
plotline3a, caplines3a, barlinecols3a = ax3.errorbar(ldaset, meanTLarge[0], yerr = np.minimum(meanTLargeRange[0], np.sqrt(meanTLarge[0]), np.divide(meanTLarge[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline3b, caplines3b, barlinecols3b = ax3.errorbar(ldaset, meanTLarge[1], yerr = np.minimum(meanTLargeRange[1], np.sqrt(meanTLarge[1]), np.divide(meanTLarge[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline3c, caplines3c, barlinecols3c = ax3.errorbar(ldaset, meanTLarge[2], yerr = np.minimum(meanTLargeRange[2], np.sqrt(meanTLarge[2]), np.divide(meanTLarge[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline3d, caplines3d, barlinecols3d = ax3.errorbar(ldaset, meanTLarge[3], yerr = np.minimum(meanTLargeRange[3], np.sqrt(meanTLarge[3]), np.divide(meanTLarge[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines3a[0].set_marker('')
caplines3b[0].set_marker('')
caplines3c[0].set_marker('')
caplines3d[0].set_marker('')
handles3, labels3 = ax3.get_legend_handles_labels()
handles3 = [h3[0] for h3 in handles3]
ax3.legend(handles3, labels3, loc = 'lower right')
ax3.set_yscale('log')
ax3.set_ylim(0.01, 20000)
ax3.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax3.set_ylabel("MSE of PRIEST-KLD")
ax3.figure.savefig("Eps_low_exp1_femnist_Ct_est_a_720.png")
plt.close()

fig, ax4 = plt.subplots(layout = 'constrained')
plotline4a, caplines4a, barlinecols4a = ax4.errorbar(ldaset, minTSmall[0], yerr = np.minimum(minTSmallRange[0], np.sqrt(minTSmall[0]), np.divide(minTSmall[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline4b, caplines4b, barlinecols4b = ax4.errorbar(ldaset, minTSmall[1], yerr = np.minimum(minTSmallRange[1], np.sqrt(minTSmall[1]), np.divide(minTSmall[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline4c, caplines4c, barlinecols4c = ax4.errorbar(ldaset, minTSmall[2], yerr = np.minimum(minTSmallRange[2], np.sqrt(minTSmall[2]), np.divide(minTSmall[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline4d, caplines4d, barlinecols4d = ax4.errorbar(ldaset, minTSmall[3], yerr = np.minimum(minTSmallRange[3], np.sqrt(minTSmall[3]), np.divide(minTSmall[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines4a[0].set_marker('')
caplines4b[0].set_marker('')
caplines4c[0].set_marker('')
caplines4d[0].set_marker('')
handles4, labels4 = ax4.get_legend_handles_labels()
handles4 = [h4[0] for h4 in handles4]
ax4.legend(handles4, labels4, loc = 'best')
ax4.set_yscale('log')
ax4.set_ylim(0.07, 20000)
ax4.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax4.set_ylabel("MSE of PRIEST-KLD")
ax4.figure.savefig("Eps_low_exp1_femnist_Ct_est_b_36.png")
plt.close()

fig, ax5 = plt.subplots(layout = 'constrained')
plotline5a, caplines5a, barlinecols5a = ax5.errorbar(ldaset, minTMid[0], yerr = np.minimum(minTMidRange[0], np.sqrt(minTMid[0]), np.divide(minTMid[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline5b, caplines5b, barlinecols5b = ax5.errorbar(ldaset, minTMid[1], yerr = np.minimum(minTMidRange[1], np.sqrt(minTMid[1]), np.divide(minTMid[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline5c, caplines5c, barlinecols5c = ax5.errorbar(ldaset, minTMid[2], yerr = np.minimum(minTMidRange[2], np.sqrt(minTMid[2]), np.divide(minTMid[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline5d, caplines5d, barlinecols5d = ax5.errorbar(ldaset, minTMid[3], yerr = np.minimum(minTMidRange[3], np.sqrt(minTMid[3]), np.divide(minTMid[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines5a[0].set_marker('')
caplines5b[0].set_marker('')
caplines5c[0].set_marker('')
caplines5d[0].set_marker('')
handles5, labels5 = ax5.get_legend_handles_labels()
handles5 = [h5[0] for h5 in handles5]
ax5.legend(handles5, labels5, loc = 'lower right')
ax5.set_yscale('log')
ax5.set_ylim(0.03, 20000)
ax5.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax5.set_ylabel("MSE of PRIEST-KLD")
ax5.figure.savefig("Eps_low_exp1_femnist_Ct_est_b_270.png")
plt.close()

fig, ax6 = plt.subplots(layout = 'constrained')
plotline6a, caplines6a, barlinecols6a = ax6.errorbar(ldaset, minTLarge[0], yerr = np.minimum(minTLargeRange[0], np.sqrt(minTLarge[0]), np.divide(minTLarge[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline6b, caplines6b, barlinecols6b = ax6.errorbar(ldaset, minTLarge[1], yerr = np.minimum(minTLargeRange[1], np.sqrt(minTLarge[1]), np.divide(minTLarge[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline6c, caplines6c, barlinecols6c = ax6.errorbar(ldaset, minTLarge[2], yerr = np.minimum(minTLargeRange[2], np.sqrt(minTLarge[2]), np.divide(minTLarge[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline6d, caplines6d, barlinecols6d = ax6.errorbar(ldaset, minTLarge[3], yerr = np.minimum(minTLargeRange[3], np.sqrt(minTLarge[3]), np.divide(minTLarge[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines6a[0].set_marker('')
caplines6b[0].set_marker('')
caplines6c[0].set_marker('')
caplines6d[0].set_marker('')
handles6, labels6 = ax6.get_legend_handles_labels()
handles6 = [h6[0] for h6 in handles6]
ax6.legend(handles6, labels6, loc = 'center right')
ax6.set_yscale('log')
ax6.set_ylim(0.1, 8000)
ax6.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax6.set_ylabel("MSE of PRIEST-KLD")
ax6.figure.savefig("Eps_low_exp1_femnist_Ct_est_b_720.png")
plt.close()

fig, ax7 = plt.subplots(layout = 'constrained')
plotline7a, caplines7a, barlinecols7a = ax7.errorbar(ldaset, maxTSmall[0], yerr = np.minimum(maxTSmallRange[0], np.sqrt(maxTSmall[0]), np.divide(maxTSmall[0], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline7b, caplines7b, barlinecols7b = ax7.errorbar(ldaset, maxTSmall[1], yerr = np.minimum(maxTSmallRange[1], np.sqrt(maxTSmall[1]), np.divide(maxTSmall[1], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline7c, caplines7c, barlinecols7c = ax7.errorbar(ldaset, maxTSmall[2], yerr = np.minimum(maxTSmallRange[2], np.sqrt(maxTSmall[2]), np.divide(maxTSmall[2], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline7d, caplines7d, barlinecols7d = ax7.errorbar(ldaset, maxTSmall[3], yerr = np.minimum(maxTSmallRange[3], np.sqrt(maxTSmall[3]), np.divide(maxTSmall[3], 2)),
                                                     uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines7a[0].set_marker('')
caplines7b[0].set_marker('')
caplines7c[0].set_marker('')
caplines7d[0].set_marker('')
handles7, labels7 = ax7.get_legend_handles_labels()
handles7 = [h7[0] for h7 in handles7]
ax7.legend(handles7, labels7, loc = 'best')
ax7.set_yscale('log')
ax7.set_ylim(1, 20000)
ax7.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax7.set_ylabel("MSE of PRIEST-KLD")
ax7.figure.savefig("Eps_low_exp1_femnist_Ct_est_c_36.png")
plt.close()

fig, ax8 = plt.subplots(layout = 'constrained')
plotline8a, caplines8a, barlinecols8a = ax8.errorbar(ldaset, maxTMid[0], yerr = np.minimum(maxTMidRange[0], np.sqrt(maxTMid[0]), np.divide(maxTMid[0], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline8b, caplines8b, barlinecols8b = ax8.errorbar(ldaset, maxTMid[1], yerr = np.minimum(maxTMidRange[1], np.sqrt(maxTMid[1]), np.divide(maxTMid[1], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline8c, caplines8c, barlinecols8c = ax8.errorbar(ldaset, maxTMid[2], yerr = np.minimum(maxTMidRange[2], np.sqrt(maxTMid[2]), np.divide(maxTMid[2], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline8d, caplines8d, barlinecols8d = ax8.errorbar(ldaset, maxTMid[3], yerr = np.minimum(maxTMidRange[3], np.sqrt(maxTMid[3]), np.divide(maxTMid[3], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines8a[0].set_marker('')
caplines8b[0].set_marker('')
caplines8c[0].set_marker('')
caplines8d[0].set_marker('')
handles8, labels8 = ax8.get_legend_handles_labels()
handles8 = [h8[0] for h8 in handles8]
ax8.legend(handles8, labels8, loc = 'best')
ax8.set_yscale('log')
ax8.set_ylim(1, 20000)
ax8.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax8.set_ylabel("MSE of PRIEST-KLD")
ax8.figure.savefig("Eps_low_exp1_femnist_Ct_est_c_270.png")
plt.close()

fig, ax9 = plt.subplots(layout = 'constrained')
plotline9a, caplines9a, barlinecols9a = ax9.errorbar(ldaset, maxTLarge[0], yerr = np.minimum(maxTLargeRange[0], np.sqrt(maxTLarge[0]), np.divide(maxTLarge[0], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'blue', marker = 'o', label = "Dist")
plotline9b, caplines9b, barlinecols9b = ax9.errorbar(ldaset, maxTLarge[1], yerr = np.minimum(maxTLargeRange[1], np.sqrt(maxTLarge[1]), np.divide(maxTLarge[1], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'green', marker = 'o', label = "TAgg")
plotline9c, caplines9c, barlinecols9c = ax9.errorbar(ldaset, maxTLarge[2], yerr = np.minimum(maxTLargeRange[2], np.sqrt(maxTLarge[2]), np.divide(maxTLarge[2], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'orange', marker = 'o', label = "Trusted")
plotline9d, caplines9d, barlinecols9d = ax9.errorbar(ldaset, maxTLarge[3], yerr = np.minimum(maxTLargeRange[3], np.sqrt(maxTLarge[3]), np.divide(maxTLarge[3], 2)),
                                                         uplims = upldaset, lolims = loldaset, color = 'red', marker = '*', label = "no privacy")
caplines9a[0].set_marker('')
caplines9b[0].set_marker('')
caplines9c[0].set_marker('')
caplines9d[0].set_marker('')
handles9, labels9 = ax9.get_legend_handles_labels()
handles9 = [h9[0] for h9 in handles9]
ax9.legend(handles9, labels9, loc = 'best')
ax9.set_yscale('log')
ax9.set_ylim(1, 10000)
ax9.set_xlabel("Value of " + "$\mathit{\u03bb}$")
ax9.set_ylabel("MSE of PRIEST-KLD")
ax9.figure.savefig("Eps_low_exp1_femnist_Ct_est_c_720.png")
plt.close()

# EXPERIMENT 2: MSE of PRIEST-KLD for each Ct
fig, ax10 = plt.subplots(layout = 'constrained')
plotline10a, caplines10a, barlinecols10a = ax10.errorbar(Ctset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'fuchsia', marker = 'o', label = "mean")
plotline10b, caplines10b, barlinecols10b = ax10.errorbar(Ctset, minEstMSE[0], yerr = np.minimum(minEstRange[0], np.sqrt(minEstMSE[0]), np.divide(minEstMSE[0], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'mediumorchid', marker = 'o', label = "min pair")
plotline10c, caplines10c, barlinecols10c = ax10.errorbar(Ctset, maxEstMSE[0], yerr = np.minimum(maxEstRange[0], np.sqrt(maxEstMSE[0]), np.divide(maxEstMSE[0], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'darkcyan', marker = 'o', label = "max pair")
caplines10a[0].set_marker('')
caplines10b[0].set_marker('')
caplines10c[0].set_marker('')
handles10, labels10 = ax10.get_legend_handles_labels()
handles10 = [h10[0] for h10 in handles10]
ax10.legend(handles10, labels10, loc = 'best')
ax10.set_yscale('log')
ax10.set_ylim(0.01, 200)
ax10.set_xlabel("Client sample size " + "$\mathit{|C_{t}|}$")
ax10.set_ylabel("MSE of PRIEST-KLD")
ax10.figure.savefig("Eps_low_exp2_femnist_Ct_est_a.png")
plt.close()

fig, ax11 = plt.subplots(layout = 'constrained')
plotline11a, caplines11a, barlinecols11a = ax11.errorbar(Ctset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'fuchsia', marker = 'o', label = "mean")
plotline11b, caplines11b, barlinecols11b = ax11.errorbar(Ctset, minEstMSE[1], yerr = np.minimum(minEstRange[1], np.sqrt(minEstMSE[1]), np.divide(minEstMSE[1], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'mediumorchid', marker = 'o', label = "min pair")
plotline11c, caplines11c, barlinecols11c = ax11.errorbar(Ctset, maxEstMSE[1], yerr = np.minimum(maxEstRange[1], np.sqrt(maxEstMSE[1]), np.divide(maxEstMSE[1], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'darkcyan', marker = 'o', label = "max pair")
caplines11a[0].set_marker('')
caplines11b[0].set_marker('')
caplines11c[0].set_marker('')
handles11, labels11 = ax11.get_legend_handles_labels()
handles11 = [h11[0] for h11 in handles11]
ax11.legend(handles11, labels11, loc = 'best')
ax11.set_yscale('log')
ax11.set_ylim(0.1, 70)
ax11.set_xlabel("Client sample size " + "$\mathit{|C_{t}|}$")
ax11.set_ylabel("MSE of PRIEST-KLD")
ax11.figure.savefig("Eps_low_exp2_femnist_Ct_est_b.png")
plt.close()

fig, ax12 = plt.subplots(layout = 'constrained')
plotline12a, caplines12a, barlinecols12a = ax12.errorbar(Ctset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'fuchsia', marker = 'o', label = "mean")
plotline12b, caplines12b, barlinecols12b = ax12.errorbar(Ctset, minEstMSE[2], yerr = np.minimum(minEstRange[2], np.sqrt(minEstMSE[2]), np.divide(minEstMSE[2], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'mediumorchid', marker = 'o', label = "min pair")
plotline12c, caplines12c, barlinecols12c = ax12.errorbar(Ctset, maxEstMSE[2], yerr = np.minimum(maxEstRange[2], np.sqrt(maxEstMSE[2]), np.divide(maxEstMSE[2], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'darkcyan', marker = 'o', label = "max pair")
caplines12a[0].set_marker('')
caplines12b[0].set_marker('')
caplines12c[0].set_marker('')
handles12, labels12 = ax12.get_legend_handles_labels()
handles12 = [h12[0] for h12 in handles12]
ax12.legend(handles12, labels12, loc = 'best')
ax12.set_yscale('log')
ax12.set_ylim(300, 20000)
ax12.set_xlabel("Client sample size " + "$\mathit{|C_{t}|}$")
ax12.set_ylabel("MSE of PRIEST-KLD")
ax12.figure.savefig("Eps_low_exp2_femnist_Ct_est_c.png")
plt.close()

fig, ax13 = plt.subplots(layout = 'constrained')
plotline13a, caplines13a, barlinecols13a = ax13.errorbar(Ctset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'blue', marker = 'o', label = "Dist")
plotline13b, caplines13b, barlinecols13b = ax13.errorbar(Ctset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'green', marker = 'o', label = "TAgg")
plotline13c, caplines13c, barlinecols13c = ax13.errorbar(Ctset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'orange', marker = 'o', label = "Trusted")
plotline13d, caplines13d, barlinecols13d = ax13.errorbar(Ctset, meanEstMSE[3], yerr = np.minimum(meanEstRange[3], np.sqrt(meanEstMSE[3]), np.divide(meanEstMSE[3], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'red', marker = '*', label = "no privacy")
caplines13a[0].set_marker('')
caplines13b[0].set_marker('')
caplines13c[0].set_marker('')
caplines13d[0].set_marker('')
handles13, labels13 = ax13.get_legend_handles_labels()
handles13 = [h13[0] for h13 in handles13]
ax13.legend(handles13, labels13, loc = 'best')
ax13.set_yscale('log')
ax13.set_ylim(0.008, 50000)
ax13.set_xlabel("Client sample size " + "$\mathit{|C_{t}|}$")
ax13.set_ylabel("MSE of PRIEST-KLD")
ax13.figure.savefig("Eps_low_exp2_femnist_Ct_est_d.png")
plt.close()

fig, ax14 = plt.subplots(layout = 'constrained')
plotline14a, caplines14a, barlinecols14a = ax14.errorbar(Ctset, minEstMSE[0], yerr = np.minimum(minEstRange[0], np.sqrt(minEstMSE[0]), np.divide(minEstMSE[0], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'blue', marker = 'o', label = "Dist")
plotline14b, caplines14b, barlinecols14b = ax14.errorbar(Ctset, minEstMSE[1], yerr = np.minimum(minEstRange[1], np.sqrt(minEstMSE[1]), np.divide(minEstMSE[1], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'green', marker = 'o', label = "TAgg")
plotline14c, caplines14c, barlinecols14c = ax14.errorbar(Ctset, minEstMSE[2], yerr = np.minimum(minEstRange[2], np.sqrt(minEstMSE[2]), np.divide(minEstMSE[2], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'orange', marker = 'o', label = "Trusted")
plotline14d, caplines14d, barlinecols14d = ax14.errorbar(Ctset, minEstMSE[3], yerr = np.minimum(minEstRange[3], np.sqrt(minEstMSE[3]), np.divide(minEstMSE[3], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'red', marker = '*', label = "no privacy")
caplines14a[0].set_marker('')
caplines14b[0].set_marker('')
caplines14c[0].set_marker('')
caplines14d[0].set_marker('')
handles14, labels14 = ax14.get_legend_handles_labels()
handles14 = [h14[0] for h14 in handles14]
ax14.legend(handles14, labels14, loc = 'center left')
ax14.set_yscale('log')
ax14.set_ylim(0.03, 50000)
ax14.set_xlabel("Client sample size " + "$\mathit{|C_{t}|}$")
ax14.set_ylabel("MSE of PRIEST-KLD")
ax14.figure.savefig("Eps_low_exp2_femnist_Ct_est_e.png")
plt.close()

fig, ax15 = plt.subplots(layout = 'constrained')
plotline15a, caplines15a, barlinecols15a = ax15.errorbar(Ctset, maxEstMSE[0], yerr = np.minimum(maxEstRange[0], np.sqrt(maxEstMSE[0]), np.divide(maxEstMSE[0], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'blue', marker = 'o', label = "Dist")
plotline15b, caplines15b, barlinecols15b = ax15.errorbar(Ctset, maxEstMSE[1], yerr = np.minimum(maxEstRange[1], np.sqrt(maxEstMSE[1]), np.divide(maxEstMSE[1], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'green', marker = 'o', label = "TAgg")
plotline15c, caplines15c, barlinecols15c = ax15.errorbar(Ctset, maxEstMSE[2], yerr = np.minimum(maxEstRange[2], np.sqrt(maxEstMSE[2]), np.divide(maxEstMSE[2], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'orange', marker = 'o', label = "Trusted")
plotline15d, caplines15d, barlinecols15d = ax15.errorbar(Ctset, maxEstMSE[3], yerr = np.minimum(maxEstRange[3], np.sqrt(maxEstMSE[3]), np.divide(maxEstMSE[3], 2)),
                                                         uplims = upTset, lolims = loTset, color = 'red', marker = '*', label = "no privacy")
caplines15a[0].set_marker('')
caplines15b[0].set_marker('')
caplines15c[0].set_marker('')
caplines15d[0].set_marker('')
handles15, labels15 = ax15.get_legend_handles_labels()
handles15 = [h15[0] for h15 in handles15]
ax15.legend(handles15, labels15, loc = 'best')
ax15.set_yscale('log')
ax15.set_ylim(2, 8000)
ax15.set_xlabel("Client sample size " + "$\mathit{|C_{t}|}$")
ax15.set_ylabel("MSE of PRIEST-KLD")
ax15.figure.savefig("Eps_low_exp2_femnist_Ct_est_f.png")
plt.close()

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