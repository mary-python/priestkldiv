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
epsset = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6]
ldaset = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
          1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
trialset = ["Dist", "TAgg", "Trusted", "no_privacy"]
ES = len(epsset)
LS = len(ldaset)
TS = len(trialset)

# to store statistics related to mean estimates
meanValue = np.zeros((TS, ES))
meanEstMSE = np.zeros((TS, ES))
meanLdaBest = np.zeros((TS, ES))
meanSmallBest = np.zeros((TS, ES))
meanMidDist = np.zeros((TS, ES))
meanMidTAgg = np.zeros((TS, ES))
meanMidTrusted = np.zeros((TS, ES))
meanPerc = np.zeros((TS, ES))
meanEpsSmall = np.zeros((TS, LS))
meanEpsMid = np.zeros((TS, LS))

meanEstRange = np.zeros((TS, ES))
meanSmallBestRange = np.zeros((TS, ES))
meanMidDistRange = np.zeros((TS, ES))
meanMidTAggRange = np.zeros((TS, ES))
meanMidTrustedRange = np.zeros((TS, ES))
meanEpsSmallRange = np.zeros((TS, LS))
meanEpsMidRange = np.zeros((TS, LS))

# related to min pairs
minValue = np.zeros((TS, ES))
minEstMSE = np.zeros((TS, ES))
minLdaBest = np.zeros((TS, ES))
minPerc = np.zeros((TS, ES))
minEstRange = np.zeros((TS, ES))
minPercRange = np.zeros((TS, ES))

# related to max pairs
maxValue = np.zeros((TS, ES))
maxEstMSE = np.zeros((TS, ES))
maxLdaBest = np.zeros((TS, ES))
maxPerc = np.zeros((TS, ES))
maxEstRange = np.zeros((TS, ES))
maxPercRange = np.zeros((TS, ES))

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
MID_INDEX = 7

for trial in range(4):
    meanfile = open(f"Data_{trialset[trial]}_eps_a.txt", "w", encoding = 'utf-8')
    minfile = open(f"Data_{trialset[trial]}_eps_b.txt", "w", encoding = 'utf-8')
    maxfile = open(f"Data_{trialset[trial]}_eps_c.txt", "w", encoding = 'utf-8')
    EPS_FREQ = 0

    for eps in epsset:
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        # temporary stores for each repeat
        tempMeanValue = np.zeros(RS)
        tempMeanEst = np.zeros(RS)
        tempMeanEstMSE = np.zeros(RS)
        tempMeanLdaBest = np.zeros(RS)
        tempMeanSmallBest = np.zeros(RS)
        tempMeanMidDist = np.zeros(RS)
        tempMeanMidTAgg = np.zeros(RS)
        tempMeanMidTrusted = np.zeros(RS)
        tempMeanPerc = np.zeros(RS)
        tempMeanEpsSmall = np.zeros((LS, RS))
        tempMeanEpsMid = np.zeros((LS, RS))
            
        tempMinValue = np.zeros(RS)
        tempMinEst = np.zeros(RS)
        tempMinEstMSE = np.zeros(RS)
        tempMinLdaBest = np.zeros(RS)
        tempMinPerc = np.zeros(RS)

        tempMaxValue = np.zeros(RS)
        tempMaxEst = np.zeros(RS)
        tempMaxEstMSE = np.zeros(RS)
        tempMaxLdaBest = np.zeros(RS)
        tempMaxPerc = np.zeros(RS)

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
                    invRatio = abs(sum(uDist[C, D]) / sum(nDist[C, D]))

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
            
                # mean across lambdas for eps = 0.05 (small)
                if EPS_FREQ == SMALL_INDEX:
                    tempMeanEpsSmall[l, rep] = meanLda[l]

                # eps = 1.5 (mid)
                if EPS_FREQ == MID_INDEX:
                    tempMeanEpsMid[l, rep] = meanLda[l]

            # find lambda that produces minimum error
            meanLdaIndex = np.argmin(meanLda)
            minLdaIndex = np.argmin(minLda)
            maxLdaIndex = np.argmin(maxLda)

            meanMinError = meanLda[meanLdaIndex]
            minMinError = minLda[minLdaIndex]
            maxMinError = maxLda[maxLdaIndex]

            # mean / min / max across clients for best lambda
            tempMeanEst[rep] = meanMinError
            tempMinEst[rep] = minMinError
            tempMaxEst[rep] = maxMinError

            # best lambda
            tempMeanLdaBest[rep] = meanLdaIndex * ldaStep
            tempMinLdaBest[rep] = minLdaIndex * ldaStep
            tempMaxLdaBest[rep] = maxLdaIndex * ldaStep

            # eps = 0.05, lambda = 0.6 (Dist)
            tempMeanSmallBest[rep] = meanLda[12]

            # eps = 1.5, lambda = 1 (Dist)
            tempMeanMidDist[rep] = meanLda[20]

            # eps = 1.5, lambda = 0.55 (TAgg)
            tempMeanMidTAgg[rep] = meanLda[11]

            # eps = 1.5, lambda = 0.4 (Trusted)
            tempMeanMidTrusted[rep] = meanLda[8]

            # "Trusted" (server adds Laplace noise term to final result)
            if trial == 2:
                lapNoise = tfp.distributions.Laplace(loc = A, scale = b1)
                meanNoise = lapNoise.sample(sample_shape = (1,))
                minNoise = lapNoise.sample(sample_shape = (1,))
                maxNoise = lapNoise.sample(sample_shape = (1,))
                meanSmallBestNoise = lapNoise.sample(sample_shape = (1,))
                meanMidDistNoise = lapNoise.sample(sample_shape = (1,))
                meanMidTAggNoise = lapNoise.sample(sample_shape = (1,))
                meanMidTrustedNoise = lapNoise.sample(sample_shape = (1,))

                # define error = squared difference between estimator and ground truth
                tempMeanEstMSE[rep] = (tempMeanEst[rep] + meanNoise - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] + minNoise - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] + maxNoise - tempMaxValue[rep])**2
                tempMeanSmallBest[rep] = (tempMeanSmallBest[rep] + meanSmallBestNoise - tempMeanValue[rep])**2
                tempMeanMidDist[rep] = (tempMeanMidDist[rep] + meanMidDistNoise - tempMeanValue[rep])**2
                tempMeanMidTAgg[rep] = (tempMeanMidTAgg[rep] + meanMidTAggNoise - tempMeanValue[rep])**2
                tempMeanMidTrusted[rep] = (tempMeanMidTrusted[rep] + meanMidTrustedNoise - tempMeanValue[rep])**2

                for l in range(LS):
        
                    # eps = 0.05 (small)
                    if EPS_FREQ == SMALL_INDEX:
                        meanSmallNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsSmall[l, rep] = (tempMeanEpsSmall[l, rep] + meanSmallNoise - tempMeanValue[rep])**2

                    # eps = 1.5 (mid)
                    if EPS_FREQ == MID_INDEX:
                        meanMidNoise = lapNoise.sample(sample_shape = (1,))
                        tempMeanEpsMid[l, rep] = (tempMeanEpsMid[l, rep] + meanMidNoise - tempMeanValue[rep])**2
        
            # clients or intermediate server already added Gaussian noise term
            else:
                tempMeanEstMSE[rep] = (tempMeanEst[rep] - tempMeanValue[rep])**2
                tempMinEstMSE[rep] = (tempMinEst[rep] - tempMinValue[rep])**2
                tempMaxEstMSE[rep] = (tempMaxEst[rep] - tempMaxValue[rep])**2
                tempMeanSmallBest[rep] = (tempMeanSmallBest[rep] - tempMeanValue[rep])**2
                tempMeanMidDist[rep] = (tempMeanMidDist[rep] - tempMeanValue[rep])**2
                tempMeanMidTAgg[rep] = (tempMeanMidTAgg[rep] - tempMeanValue[rep])**2
                tempMeanMidTrusted[rep] = (tempMeanMidTrusted[rep] - tempMeanValue[rep])**2

                for l in range(LS):
        
                    # eps = 0.05 (small)
                    if EPS_FREQ == SMALL_INDEX:
                        tempMeanEpsSmall[l, rep] = (tempMeanEpsSmall[l, rep] - tempMeanValue[rep])**2

                    # eps = 1.5 (mid)
                    if EPS_FREQ == MID_INDEX:
                        tempMeanEpsMid[l, rep] = (tempMeanEpsMid[l, rep] - tempMeanValue[rep])**2
            
            # compute % of noise vs ground truth and extract MSE of noise for Theorem 4.4
            if trial == 0:
                tempMeanPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMeanValue[rep])))*100
                tempMinPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMinValue[rep])))*100
                tempMaxPerc[rep] = float(abs(np.array(sum(startNoise)) / (np.array(sum(startNoise)) + tempMaxValue[rep])))*100      

            if trial == 1:
                tempMinPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMinValue[rep]))*100
                tempMaxPerc[rep] = abs((np.sum(meanLdaNoise)) / (np.sum(meanLdaNoise) + tempMaxValue[rep]))*100
            
            if trial == 2:
                tempMinPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMinValue[rep])))*100
                tempMaxPerc[rep] = float(abs(np.array(meanNoise) / (np.array(meanNoise) + tempMaxValue[rep])))*100
            
            SEED_FREQ = SEED_FREQ + 1

        # compute mean of repeats
        meanValue[trial, EPS_FREQ] = np.mean(tempMeanValue)
        meanEstMSE[trial, EPS_FREQ] = np.mean(tempMeanEstMSE)
        meanLdaBest[trial, EPS_FREQ] = np.mean(tempMeanLdaBest)
        meanSmallBest[trial, EPS_FREQ] = np.mean(tempMeanSmallBest)
        meanMidDist[trial, EPS_FREQ] = np.mean(tempMeanMidDist)
        meanMidTAgg[trial, EPS_FREQ] = np.mean(tempMeanMidTAgg)
        meanMidTrusted[trial, EPS_FREQ] = np.mean(tempMeanMidTrusted)
        meanPerc[trial, EPS_FREQ] = np.mean(tempMeanPerc)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                meanEpsSmall[trial, l] = np.mean(tempMeanEpsSmall[l])
            if EPS_FREQ == MID_INDEX:
                meanEpsMid[trial, l] = np.mean(tempMeanEpsMid[l])

        minValue[trial, EPS_FREQ] = np.mean(tempMinValue)
        minEstMSE[trial, EPS_FREQ] = np.mean(tempMinEstMSE)
        minLdaBest[trial, EPS_FREQ] = np.mean(tempMinLdaBest)
        minPerc[trial, EPS_FREQ] = np.mean(tempMinPerc)

        maxValue[trial, EPS_FREQ] = np.mean(tempMaxValue)
        maxEstMSE[trial, EPS_FREQ] = np.mean(tempMaxEstMSE)
        maxLdaBest[trial, EPS_FREQ] = np.mean(tempMaxLdaBest)
        maxPerc[trial, EPS_FREQ] = np.mean(tempMaxPerc)

        # compute standard deviation of repeats
        meanEstRange[trial, EPS_FREQ] = np.std(tempMeanEstMSE)
        meanSmallBestRange[trial, EPS_FREQ] = np.std(tempMeanSmallBest)
        meanMidDistRange[trial, EPS_FREQ] = np.std(tempMeanMidDist)
        meanMidTAggRange[trial, EPS_FREQ] = np.std(tempMeanMidTAgg)
        meanMidTrustedRange[trial, EPS_FREQ] = np.std(tempMeanMidTrusted)

        for l in range(LS):
            if EPS_FREQ == SMALL_INDEX:
                meanEpsSmallRange[trial, l] = np.std(tempMeanEpsSmall[l])
            if EPS_FREQ == MID_INDEX:
                meanEpsMidRange[trial, l] = np.std(tempMeanEpsMid[l])

        minEstRange[trial, EPS_FREQ] = np.std(tempMinEstMSE)
        minPercRange[trial, EPS_FREQ] = np.std(tempMinPerc)
        maxEstRange[trial, EPS_FREQ] = np.std(tempMaxEstMSE)
        maxPercRange[trial, EPS_FREQ] = np.std(tempMaxPerc)

        # write statistics on data files
        if eps == epsset[0]:
            meanfile.write(f"FEMNIST: Eps = {eps}\n")
            minfile.write(f"FEMNIST: Eps = {eps}\n")
            maxfile.write(f"FEMNIST: Eps = {eps}\n")
        else:
            meanfile.write(f"\nEps = {eps}\n")
            minfile.write(f"\nEps = {eps}\n")
            maxfile.write(f"\nEps = {eps}\n")

        meanfile.write(f"\nMean MSE: {round(meanEstMSE[trial, EPS_FREQ], 2)}\n")
        meanfile.write(f"Best Lambda: {round(meanLdaBest[trial, EPS_FREQ], 2)}\n")
        meanfile.write(f"Ground Truth: {round(meanValue[trial, EPS_FREQ], 2)}\n")
        meanfile.write(f"Noise: {np.round(meanPerc[trial, EPS_FREQ], 2)}%\n")
        
        minfile.write(f"\nMin MSE: {round(minEstMSE[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Best Lambda: {round(minLdaBest[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Ground Truth: {round(minValue[trial, EPS_FREQ], 2)}\n")
        minfile.write(f"Noise: {np.round(minPerc[trial, EPS_FREQ], 2)}%\n")
        
        maxfile.write(f"\nMax MSE: {round(maxEstMSE[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Best Lambda: {round(maxLdaBest[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Ground Truth: {round(maxValue[trial, EPS_FREQ], 2)}\n")
        maxfile.write(f"Noise: {np.round(maxPerc[trial, EPS_FREQ], 2)}%\n")

        EPS_FREQ = EPS_FREQ + 1

# EXPERIMENT 1: MSE of PRIEST-KLD for each epsilon
plt.errorbar(epsset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)), color = 'blueviolet', marker = 'o', label = "mean")
plt.errorbar(epsset, minEstMSE[0], yerr = np.minimum(minEstRange[0], np.sqrt(minEstMSE[0]), np.divide(minEstMSE[0], 2)), color = 'lime', marker = 'o', label = "min pair")
plt.errorbar(epsset, maxEstMSE[0], yerr = np.minimum(maxEstRange[0], np.sqrt(maxEstMSE[0]), np.divide(maxEstMSE[0], 2)), color = 'gold', marker = 'o', label = "max pair")
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.ylim(0.1, 40)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_eps_est_a.png")
plt.clf()

plt.errorbar(epsset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)), color = 'blueviolet', marker = 'o', label = "mean")
plt.errorbar(epsset, minEstMSE[1], yerr = np.minimum(minEstRange[1], np.sqrt(minEstMSE[1]), np.divide(minEstMSE[1], 2)), color = 'lime', marker = 'o', label = "min pair")
plt.errorbar(epsset, maxEstMSE[1], yerr = np.minimum(maxEstRange[1], np.sqrt(maxEstMSE[1]), np.divide(maxEstMSE[1], 2)), color = 'gold', marker = 'o', label = "max pair")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.3, 30000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_eps_est_b.png")
plt.clf()

plt.errorbar(epsset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)), color = 'blueviolet', marker = 'o', label = "mean")
plt.errorbar(epsset, minEstMSE[2], yerr = np.minimum(minEstRange[2], np.sqrt(minEstMSE[2]), np.divide(minEstMSE[2], 2)), color = 'lime', marker = 'o', label = "min pair")
plt.errorbar(epsset, maxEstMSE[2], yerr = np.minimum(maxEstRange[2], np.sqrt(maxEstMSE[2]), np.divide(maxEstMSE[2], 2)), color = 'gold', marker = 'o', label = "max pair")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.4, 6000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_eps_est_c.png")
plt.clf()

plt.errorbar(epsset, meanEstMSE[0], yerr = np.minimum(meanEstRange[0], np.sqrt(meanEstMSE[0]), np.divide(meanEstMSE[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanEstMSE[1], yerr = np.minimum(meanEstRange[1], np.sqrt(meanEstMSE[1]), np.divide(meanEstMSE[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanEstMSE[2], yerr = np.minimum(meanEstRange[2], np.sqrt(meanEstMSE[2]), np.divide(meanEstMSE[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.3, 30000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp1_femnist_eps_est_d.png")
plt.clf()

# EXPERIMENT 2: % of noise vs ground truth for each epsilon (min / max pairs)
plt.errorbar(epsset, minPerc[0], yerr = np.minimum(minPercRange[0], np.sqrt(minPerc[0]), np.divide(minPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, minPerc[1], yerr = np.minimum(minPercRange[1], np.sqrt(minPerc[1]), np.divide(minPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, minPerc[2], yerr = np.minimum(minPercRange[2], np.sqrt(minPerc[2]), np.divide(minPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([0.1, 1, 10, 100, 700])
plt.ylim(0.1, 700)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("Noise (%)")
plt.savefig("Exp2_femnist_eps_perc_a.png")
plt.clf()

plt.errorbar(epsset, maxPerc[0], yerr = np.minimum(maxPercRange[0], np.sqrt(maxPerc[0]), np.divide(maxPerc[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, maxPerc[1], yerr = np.minimum(maxPercRange[1], np.sqrt(maxPerc[1]), np.divide(maxPerc[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, maxPerc[2], yerr = np.minimum(maxPercRange[2], np.sqrt(maxPerc[2]), np.divide(maxPerc[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.yticks([0.1, 1, 10, 100])
plt.ylim(0.02, 400)
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("Noise (%)")
plt.savefig("Exp2_femnist_eps_perc_b.png")
plt.clf()

# EXPERIMENT 3: MSE of PRIEST-KLD for fixed epsilons (0.05, 1.5)
plt.errorbar(ldaset, meanEpsSmall[0], yerr = np.minimum(meanEpsSmallRange[0], np.sqrt(meanEpsSmall[0]), np.divide(meanEpsSmall[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsSmall[1], yerr = np.minimum(meanEpsSmallRange[1], np.sqrt(meanEpsSmall[1]), np.divide(meanEpsSmall[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsSmall[2], yerr = np.minimum(meanEpsSmallRange[2], np.sqrt(meanEpsSmall[2]), np.divide(meanEpsSmall[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsSmall[3], yerr = np.minimum(meanEpsSmallRange[3], np.sqrt(meanEpsSmall[3]), np.divide(meanEpsSmall[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.ylim(0.005, 10000)
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_eps_est_0.05.png")
plt.clf()

plt.errorbar(ldaset, meanEpsMid[0], yerr = np.minimum(meanEpsMidRange[0], np.sqrt(meanEpsMid[0]), np.divide(meanEpsMid[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(ldaset, meanEpsMid[1], yerr = np.minimum(meanEpsMidRange[1], np.sqrt(meanEpsMid[1]), np.divide(meanEpsMid[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(ldaset, meanEpsMid[2], yerr = np.minimum(meanEpsMidRange[2], np.sqrt(meanEpsMid[2]), np.divide(meanEpsMid[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.errorbar(ldaset, meanEpsMid[3], yerr = np.minimum(meanEpsMidRange[3], np.sqrt(meanEpsMid[3]), np.divide(meanEpsMid[3], 2)), color = 'red', marker = '*', label = "no privacy")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.03, 70)
plt.xlabel("Value of " + "$\mathit{\u03bb}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp3_femnist_eps_est_1.5.png")
plt.clf()

# EXPERIMENT 4: MSE of PRIEST-KLD for best lambdas extracted from experiment 3
plt.errorbar(epsset, meanSmallBest[0], yerr = np.minimum(meanSmallBestRange[0], np.sqrt(meanSmallBest[0]), np.divide(meanSmallBest[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanSmallBest[1], yerr = np.minimum(meanSmallBestRange[1], np.sqrt(meanSmallBest[1]), np.divide(meanSmallBest[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanSmallBest[2], yerr = np.minimum(meanSmallBestRange[2], np.sqrt(meanSmallBest[2]), np.divide(meanSmallBest[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.03, 5000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_femnist_eps_best_0.05.png")
plt.clf()

plt.errorbar(epsset, meanMidDist[0], yerr = np.minimum(meanMidDistRange[0], np.sqrt(meanMidDist[0]), np.divide(meanMidDist[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanMidDist[1], yerr = np.minimum(meanMidDistRange[1], np.sqrt(meanMidDist[1]), np.divide(meanMidDist[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanMidDist[2], yerr = np.minimum(meanMidDistRange[2], np.sqrt(meanMidDist[2]), np.divide(meanMidDist[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.03, 6000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_femnist_eps_best_1.5_a.png")
plt.clf()

plt.errorbar(epsset, meanMidTAgg[0], yerr = np.minimum(meanMidTAggRange[0], np.sqrt(meanMidTAgg[0]), np.divide(meanMidTAgg[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanMidTAgg[1], yerr = np.minimum(meanMidTAggRange[1], np.sqrt(meanMidTAgg[1]), np.divide(meanMidTAgg[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanMidTAgg[2], yerr = np.minimum(meanMidTAggRange[2], np.sqrt(meanMidTAgg[2]), np.divide(meanMidTAgg[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.03, 6000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_femnist_eps_best_1.5_b.png")
plt.clf()

plt.errorbar(epsset, meanMidTrusted[0], yerr = np.minimum(meanMidTrustedRange[0], np.sqrt(meanMidTrusted[0]), np.divide(meanMidTrusted[0], 2)), color = 'blue', marker = 'o', label = "Dist")
plt.errorbar(epsset, meanMidTrusted[1], yerr = np.minimum(meanMidTrustedRange[1], np.sqrt(meanMidTrusted[1]), np.divide(meanMidTrusted[1], 2)), color = 'green', marker = 'o', label = "TAgg")
plt.errorbar(epsset, meanMidTrusted[2], yerr = np.minimum(meanMidTrustedRange[2], np.sqrt(meanMidTrusted[2]), np.divide(meanMidTrusted[2], 2)), color = 'orange', marker = 'o', label = "Trusted")
plt.legend(loc = 'best')
plt.yscale('log')
plt.ylim(0.03, 6000)
plt.xlabel("Value of " + "$\mathit{\u03b5}$")
plt.ylabel("MSE of PRIEST-KLD")
plt.savefig("Exp4_femnist_eps_best_1.5_c.png")
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