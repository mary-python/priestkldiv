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

# investigate samples from approx 1% to approx 20% of writers
Tset = [36, 72, 108, 144, 180, 225, 270, 360, 450, 540, 600, 660]
ES = len(Tset)
INDEX_FREQ = 0

for T in Tset:

    # randomly sample 5% of writers without replacement
    sampledWriters = np.random.choice(numWriters, T, replace = False)
    totalDigits = np.zeros(10, dtype = int)

    # compute the frequecy of each digit
    print("\nPreprocessing images...")
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

    print("Creating probability distributions...")

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
    EPS = 0.1
    DTA = 0.1
    A = 0
    R = 10

    # list of the trials that will be run
    trialset = ["mid_lap", "mid_lap_mc", "mid_gauss", "mid_gauss_mc", "end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc"]
    TS = len(trialset)

    # stores for sum, min/max distributions and lambda
    minSum = np.zeros((TS, ES, R))
    minPairEst = np.zeros((TS, ES, R))
    maxPairEst = np.zeros((TS, ES, R))
    sumLambda = np.zeros((TS, ES, R))
    minPairLambda = np.zeros((TS, ES, R))
    maxPairLambda = np.zeros((TS, ES, R))

    aSum = np.zeros((TS, ES))
    aPairEst = np.zeros((TS, ES))
    bPairEst = np.zeros((TS, ES))
    aLambda = np.zeros((TS, ES))
    aPairLambda = np.zeros((TS, ES))
    bPairLambda = np.zeros((TS, ES))

    # stores for ranking preservation analysis
    kPercSmall = np.zeros((TS, ES, R))
    kPercLarge = np.zeros((TS, ES, R))
    sumPercSmall = np.zeros((TS, ES, R))
    sumPercLarge = np.zeros((TS, ES, R))
    minPercSmall = np.zeros((TS, ES, R))
    minPercLarge = np.zeros((TS, ES, R))
    maxPercSmall = np.zeros((TS, ES, R))
    maxPercLarge = np.zeros((TS, ES, R))

    aPercSmall = np.zeros((TS, ES))
    aPercLarge = np.zeros((TS, ES))
    bPercSmall = np.zeros((TS, ES))
    bPercLarge = np.zeros((TS, ES))
    cPercSmall = np.zeros((TS, ES))
    cPercLarge = np.zeros((TS, ES))
    dPercSmall = np.zeros((TS, ES))
    dPercLarge = np.zeros((TS, ES))

    # for trial in range(8):
    for trial in range(4):
        print(f"\nTrial {trial + 1}: {trialset[trial]}")

        for rep in range(R):
            print(f"T = {T}: trial {trial + 1}, repeat = {rep + 1}...")

            # stores for exact and noisy unknown distributions
            uDist = np.zeros((10, 10, U))
            nDist = np.zeros((10, 10, E, R))
            uList = []
            uCDList = []         

            # stores for ratio between unknown and known distributions
            rList = []
            kList = []
            kCDList = []

            # stores for binary search
            zeroUList = []
            zeroCDList = []
            oneUList = []
            oneCDList = []
            halfUList = []
            halfCDList = []

            # option 1a: baseline case
            if trial % 2 == 0:
                b1 = log(2) / EPS

            # option 1b: Monte Carlo estimate
            else:
                b1 = (1 + log(2)) / EPS

            b2 = (2*((log(1.25))/DTA)*b1) / EPS

            # option 2a: add Laplace noise
            if trial % 4 == 0 or trial % 4 == 1:
                noiseLG = tfp.distributions.Laplace(loc = A, scale = b1)
        
            # option 2b: add Gaussian noise
            else:
                noiseLG = tfp.distributions.Normal(loc = A, scale = b2)

            def unbias_est(lda, rlist, klist, ulist, cdlist):
                """Compute sum of unbiased estimators corresponding to all pairs."""
                count = 1

                for row in range(0, len(rlist)):
                    uest = ((lda * (rlist[row] - 1)) - log(rlist[row])) / T1
          
                    # option 3b: add noise at end
                    if trial >= 4:
                        err = noiseLG.sample(sample_shape = (1,)).numpy()[0]
                    else:
                        err = 0.0

                    # add noise to unknown distribution estimator then compare to known distribution
                    err = abs(err + uest - klist[row])

                    if err != 0.0:
                        ulist.append(err)

                        c = count // 10
                        d = count % 10

                        cdlist.append((c, d))

                        if c == d + 1:
                            count = count + 2
                        else:
                            count = count + 1

                return sum(ulist)
    
            def min_max(lda, rlist, klist, ulist, cdlist, mp):
                """Compute unbiased estimator corresponding to min or max pair."""
                count = 1

                for row in range(0, len(rlist)):
                    uest = ((lda * (rlist[row] - 1)) - log(rlist[row])) / T1

                    # option 3b: add noise at end
                    if trial >= 4:
                        err = noiseLG.sample(sample_shape = (1,)).numpy()[0]
                    else:
                        err = 0.0

                    # add noise to unknown distribution estimator then compare to known distribution
                    err = abs(err + uest - klist[row])

                    if err != 0.0:
                        ulist.append(err)

                        c = count // 10
                        d = count % 10

                        cdlist.append((c, d))

                        if c == d + 1:
                            count = count + 2
                        else:
                            count = count + 1

                mi = cdlist.index(mp)
                return ulist[mi]

            # for each comparison digit compute exact and noisy unknown distributions for all digits
            for C in range(0, 10):
                for D in range(0, 10):

                    for i in range(0, U):
                        uDist[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

                    for j in range(0, E):
                        nDist[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

                        # option 3a: add noise in middle
                        if trial < 4:
                            nDist[C, D, j] = nDist[C, D, j] + noiseLG.sample(sample_shape = (1,))

                    # eliminate all zero values when digits are identical
                    if sum(uDist[C, D]) != 0.0:
                        uList.append(sum(uDist[C, D]))
                        uCDList.append((C, D))

                    # compute ratio between exact and noisy unknown distributions
                    ratio = abs(sum(nDist[C, D, j]) / sum(uDist[C, D]))

                    # eliminate all divide by zero errors
                    if ratio != 0.0 and sum(uDist[C, D]) != 0.0:
                        rList.append(ratio)

                        # compute known distribution
                        kDist = abs(sum(nDist[C, D, j]) * log(ratio))
                        kList.append(kDist)
                        kCDList.append((C, D))

                        # wait until final digit pair (9, 8) to analyse exact unknown distribution list
                        if C == 9 and D == 8:

                            low = 0
                            high = 10
                            mid = 5

                            # compute unbiased estimators with lambda 0, 1, 0.5 then binary search
                            lowSum = unbias_est(low, rList, kList, zeroUList, zeroCDList)
                            highSum = unbias_est(high, rList, kList, oneUList, oneCDList)
                            minSum[trial, INDEX_FREQ, rep] = unbias_est(mid, rList, kList, halfUList, halfCDList)

                            # tolerance between binary search limits always gets small enough
                            while abs(high - low) > 0.00000001:

                                lowUList = []
                                lowCDList = []
                                highUList = []
                                highCDList = []
                                sumUList = []
                                sumCDList = []

                                lowSum = unbias_est(low, rList, kList, lowUList, lowCDList)
                                highSum = unbias_est(high, rList, kList, highUList, highCDList)

                                # reduce / increase binary search limit depending on absolute value
                                if abs(lowSum) < abs(highSum):
                                    high = mid
                                else:
                                    low = mid

                                # set new midpoint
                                mid = (0.5*abs((high - low))) + low
                                minSum[trial, INDEX_FREQ, rep] = unbias_est(mid, rList, kList, sumUList, sumCDList)

                            sumLambda[trial, INDEX_FREQ, rep] = mid

                            # extract min pair by absolute value of exact unknown distribution
                            absUList = [abs(ul) for ul in uList]
                            minUList = sorted(absUList)
                            minAbs = minUList[0]
                            minIndex = uList.index(minAbs)
                            minPair = uCDList[minIndex]
                            MIN_COUNT = 1

                            # if min pair is not in lambda 0.5 list then get next smallest
                            while minPair not in halfCDList:        
                                minAbs = minUList[MIN_COUNT]
                                minIndex = uList.index(minAbs)
                                minPair = uCDList[minIndex]
                                MIN_COUNT = MIN_COUNT + 1

                            midMinIndex = halfCDList.index(minPair)
                            minPairEst[trial, INDEX_FREQ, rep] = halfUList[midMinIndex]

                            low = 0
                            high = 10
                            mid = 5

                            # find optimal lambda for min pair
                            while abs(high - low) > 0.00000001:

                                lowUList = []
                                lowCDList = []
                                highUList = []
                                highCDList = []
                                minUList = []
                                minCDList = []

                                lowMinUL = min_max(low, rList, kList, lowUList, lowCDList, minPair)
                                highMinUL = min_max(high, rList, kList, highUList, highCDList, minPair)

                                if abs(lowMinUL) < abs(highMinUL):
                                    high = mid
                                else:
                                    low = mid

                                mid = (0.5*abs((high - low))) + low
                                minPairEst[trial, INDEX_FREQ, rep] = min_max(mid, rList, kList, minUList, minCDList, minPair)

                            minPairLambda[trial, INDEX_FREQ, rep] = mid

                            # extract max pair by reversing unknown distribution list
                            maxUList = sorted(absUList, reverse = True)
                            maxAbs = maxUList[0]
                            maxIndex = uList.index(maxAbs)
                            maxPair = uCDList[maxIndex]
                            MAX_COUNT = 1

                            # if max pair is not in lambda 0.5 list then get next largest
                            while maxPair not in halfCDList:        
                                maxAbs = maxUList[MAX_COUNT]
                                maxIndex = uList.index(maxAbs)
                                maxPair = uCDList[maxIndex]
                                MAX_COUNT = MAX_COUNT + 1

                            midMaxIndex = halfCDList.index(maxPair)
                            maxPairEst[trial, INDEX_FREQ, rep] = halfUList[midMaxIndex]

                            low = 0
                            high = 10
                            mid = 5

                            # find optimal lambda for max pair
                            while abs(high - low) > 0.00000001:

                                lowUList = []
                                lowCDList = []
                                highUList = []
                                highCDList = []
                                maxUList = []
                                maxCDList = []

                                lowMaxUL = min_max(low, rList, kList, lowUList, lowCDList, maxPair)
                                highMaxUL = min_max(high, rList, kList, highUList, highCDList, maxPair)
                
                                if abs(lowMaxUL) < abs(highMaxUL):
                                    high = mid
                                else:
                                    low = mid

                                mid = (0.5*(abs(high - low))) + low
                                maxPairEst[trial, INDEX_FREQ, rep] = min_max(mid, rList, kList, maxUList, maxCDList, maxPair)

                            maxPairLambda[trial, INDEX_FREQ, rep] = mid

            def rank_pres(bin, oud, okd):
                """Do smallest/largest 10% in unknown distribution remain in smaller/larger half of estimator?"""
                rows = 90
                num = 0

                if bin == 0: 
                    udict = list(oud.values())[0 : int(rows / 10)]
                    kdict = list(okd.values())[0 : int(rows / 2)]
                else:
                    udict = list(oud.values())[int(9*(rows / 10)) : rows]
                    kdict = list(okd.values())[int(rows / 2) : rows]

                for ud in udict:
                    for kd in kdict:    
                        if kd == ud:
                            num = num + 1

                return 100*(num / int(rows/10))

            uDict = dict(zip(uList, uCDList))
            oUDict = OrderedDict(sorted(uDict.items()))

            # unknown distribution is identical for all Ts, trials and repeats
            if T == 36 and trial == 0 and rep == 0:
                orderfile = open("femnist_unknown_dist_in_order.txt", "w", encoding = 'utf-8')
                orderfile.write("FEMNIST: Unknown Distribution In Order\n")
                orderfile.write("Smaller corresponds to more similar digits\n\n")

                for i in oUDict:
                    orderfile.write(f"{i} : {oUDict[i]}\n")

            # compute ranking preservation statistics for each repeat
            kDict = dict(zip(kList, kCDList))
            oKDict = OrderedDict(sorted(kDict.items()))
            kPercSmall[trial, INDEX_FREQ, rep] = rank_pres(0, oUDict, oKDict)
            kPercLarge[trial, INDEX_FREQ, rep] = rank_pres(1, oUDict, oKDict)

            sumUDict = dict(zip(sumUList, sumCDList))
            sumOUDict = OrderedDict(sorted(sumUDict.items()))
            sumPercSmall[trial, INDEX_FREQ, rep] = rank_pres(0, oUDict, sumOUDict)
            sumPercLarge[trial, INDEX_FREQ, rep] = rank_pres(1, oUDict, sumOUDict)

            minUDict = dict(zip(minUList, minCDList))
            minOUDict = OrderedDict(sorted(minUDict.items()))
            minPercSmall[trial, INDEX_FREQ, rep] = rank_pres(0, oUDict, minOUDict)
            minPercLarge[trial, INDEX_FREQ, rep] = rank_pres(1, oUDict, minOUDict)

            maxUDict = dict(zip(maxUList, maxCDList))
            maxOUDict = OrderedDict(sorted(maxUDict.items()))
            maxPercSmall[trial, INDEX_FREQ, rep] = rank_pres(0, oUDict, maxOUDict)
            maxPercLarge[trial, INDEX_FREQ, rep] = rank_pres(1, oUDict, maxOUDict)
        
        # sum up repeats for all the main statistics
        aLambda[trial, INDEX_FREQ] = fmean(sumLambda[trial, INDEX_FREQ])
        aSum[trial, INDEX_FREQ] = fmean(minSum[trial, INDEX_FREQ])
        print(f"\naLambda: {aLambda[trial, INDEX_FREQ]}")
        print(f"sumLambda: {sumLambda[trial, INDEX_FREQ]}")
        print(f"aSum: {aSum[trial, INDEX_FREQ]}")
        print(f"minSum: {minSum[trial, INDEX_FREQ]}")

        aPairLambda[trial, INDEX_FREQ] = fmean(minPairLambda[trial, INDEX_FREQ])
        aPairEst[trial, INDEX_FREQ] = fmean(minPairEst[trial, INDEX_FREQ])
        bPairLambda[trial, INDEX_FREQ] = fmean(maxPairLambda[trial, INDEX_FREQ])
        bPairEst[trial, INDEX_FREQ] = fmean(maxPairEst[trial, INDEX_FREQ])

        aPercSmall[trial, INDEX_FREQ] = fmean(kPercSmall[trial, INDEX_FREQ])
        aPercLarge[trial, INDEX_FREQ] = fmean(kPercLarge[trial, INDEX_FREQ])
        bPercSmall[trial, INDEX_FREQ] = fmean(sumPercSmall[trial, INDEX_FREQ])
        bPercLarge[trial, INDEX_FREQ] = fmean(sumPercLarge[trial, INDEX_FREQ])
        cPercSmall[trial, INDEX_FREQ] = fmean(minPercSmall[trial, INDEX_FREQ])
        cPercLarge[trial, INDEX_FREQ] = fmean(minPercLarge[trial, INDEX_FREQ])
        dPercSmall[trial, INDEX_FREQ] = fmean(maxPercSmall[trial, INDEX_FREQ])
        dPercLarge[trial, INDEX_FREQ] = fmean(maxPercLarge[trial, INDEX_FREQ])

        statsfile = open(f"femnist_{trialset[trial]}_noise_t_{T}.txt", "w", encoding = 'utf-8')
        statsfile.write(f"FEMNIST: Laplace Noise in Middle, no Monte Carlo, T = {T}\n")
        statsfile.write(f"Optimal Lambda {round(aLambda[trial, INDEX_FREQ], 4)} for Sum {round(aSum[trial, INDEX_FREQ], 4)}\n\n")

        statsfile.write(f"Digit Pair with Min Exact Unknown Dist: {minPair}\n")
        statsfile.write(f"Optimal Lambda {round(aPairLambda[trial, INDEX_FREQ], 4)} for Estimate {round(aPairEst[trial, INDEX_FREQ], 4)}\n\n")

        statsfile.write(f"Digit Pair with Max Exact Unknown Dist: {maxPair}\n")
        statsfile.write(f"Optimal Lambda {round(bPairLambda[trial, INDEX_FREQ], 4)} for Estimate {round(bPairEst[trial, INDEX_FREQ], 4)}\n\n")

        statsfile.write(f"Smallest 10% exact unknown dist -> smaller half unknown dist ranking: {round(aPercSmall[trial, INDEX_FREQ], 1)}%\n")
        statsfile.write(f"Largest 10% exact unknown dist -> larger half unknown dist ranking: {round(aPercLarge[trial, INDEX_FREQ], 1)}%\n\n")
        
        statsfile.write(f"Smallest 10% exact unknown dist -> smaller half sum ranking: {round(bPercSmall[trial, INDEX_FREQ], 1)}%\n")
        statsfile.write(f"Largest 10% exact unknown dist -> larger half sum ranking: {round(bPercLarge[trial, INDEX_FREQ], 1)}%\n\n")

        statsfile.write(f"Smallest 10% exact unknown dist -> smaller half min pair ranking: {round(cPercSmall[trial, INDEX_FREQ], 1)}%\n")
        statsfile.write(f"Largest 10% exact unknown dist -> larger half min pair ranking: {round(cPercLarge[trial, INDEX_FREQ], 1)}%\n\n")

        statsfile.write(f"Smallest 10% exact unknown dist -> smaller half max pair ranking: {round(dPercSmall[trial, INDEX_FREQ], 1)}%\n")
        statsfile.write(f"Largest 10% exact unknown dist -> larger half max pair ranking: {round(dPercLarge[trial, INDEX_FREQ], 1)}%\n\n")

    INDEX_FREQ = INDEX_FREQ + 1

# plot lambdas for each T
print(f"\naLambda: {aLambda[0]}")
plt.errorbar(Tset, aLambda[0], yerr = np.std(aLambda[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap')
plt.errorbar(Tset, aLambda[1], yerr = np.std(aLambda[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc')
plt.errorbar(Tset, aLambda[2], yerr = np.std(aLambda[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, aLambda[3], yerr = np.std(aLambda[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc')
# plt.errorbar(Tset, aLambda[4], yerr = np.std(aLambda[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap')
# plt.errorbar(Tset, aLambda[5], yerr = np.std(aLambda[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc')
# plt.errorbar(Tset, aLambda[6], yerr = np.std(aLambda[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss')
# plt.errorbar(Tset, aLambda[7], yerr = np.std(aLambda[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How T affects lambda (sum)")
plt.savefig("Femnist_t_mid_lambda_sum.png")
plt.clf()

plt.errorbar(Tset, aPairLambda[0], yerr = np.std(aPairLambda[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: min')
plt.errorbar(Tset, bPairLambda[0], yerr = np.std(aPairLambda[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: max')
plt.errorbar(Tset, aPairLambda[1], yerr = np.std(aPairLambda[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: min')
plt.errorbar(Tset, bPairLambda[1], yerr = np.std(aPairLambda[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: max')
plt.errorbar(Tset, aPairLambda[2], yerr = np.std(aPairLambda[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: min')
plt.errorbar(Tset, bPairLambda[2], yerr = np.std(aPairLambda[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: max')
plt.errorbar(Tset, aPairLambda[3], yerr = np.std(aPairLambda[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: min')
plt.errorbar(Tset, bPairLambda[3], yerr = np.std(aPairLambda[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: max')
# plt.errorbar(Tset, aPairLambda[4], yerr = np.std(aPairLambda[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: min')
# plt.errorbar(Tset, bPairLambda[4], yerr = np.std(aPairLambda[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: max')
# plt.errorbar(Tset, aPairLambda[5], yerr = np.std(aPairLambda[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: min')
# plt.errorbar(Tset, bPairLambda[5], yerr = np.std(aPairLambda[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: max')
# plt.errorbar(Tset, aPairLambda[6], yerr = np.std(aPairLambda[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: min')
# plt.errorbar(Tset, bPairLambda[6], yerr = np.std(aPairLambda[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: max')
# plt.errorbar(Tset, aPairLambda[7], yerr = np.std(aPairLambda[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: min')
# plt.errorbar(Tset, bPairLambda[7], yerr = np.std(aPairLambda[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How T affects lambda (min/max pair)")
plt.savefig("Femnist_t_mid_lambda_min_max.png")
plt.clf()

# plot sum / estimates for each T
print(f"aSum: {aSum[0]}")
plt.errorbar(Tset, aSum[0], yerr = np.std(aSum[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap')
plt.errorbar(Tset, aSum[1], yerr = np.std(aSum[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc')
plt.errorbar(Tset, aSum[2], yerr = np.std(aSum[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, aSum[3], yerr = np.std(aSum[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc')
# plt.errorbar(Tset, aSum[4], yerr = np.std(aSum[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap')
# plt.errorbar(Tset, aSum[5], yerr = np.std(aSum[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc')
# plt.errorbar(Tset, aSum[6], yerr = np.std(aSum[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss')
# plt.errorbar(Tset, aSum[7], yerr = np.std(aSum[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (sum)")
plt.title("How T affects error of unbiased estimator (sum)")
plt.savefig("Femnist_t_mid_est_sum.png")
plt.clf()

plt.errorbar(Tset, aPairEst[0], yerr = np.std(aPairEst[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: min')
plt.errorbar(Tset, bPairEst[0], yerr = np.std(bPairEst[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: max')
plt.errorbar(Tset, aPairEst[1], yerr = np.std(aPairEst[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: min')
plt.errorbar(Tset, bPairEst[1], yerr = np.std(bPairEst[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: max')
plt.errorbar(Tset, aPairEst[2], yerr = np.std(aPairEst[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: min')
plt.errorbar(Tset, bPairEst[2], yerr = np.std(bPairEst[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: max')
plt.errorbar(Tset, aPairEst[3], yerr = np.std(aPairEst[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: min')
plt.errorbar(Tset, bPairEst[3], yerr = np.std(bPairEst[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: max')
# plt.errorbar(Tset, aPairEst[4], yerr = np.std(aPairEst[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: min')
# plt.errorbar(Tset, bPairEst[4], yerr = np.std(bPairEst[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: max')
# plt.errorbar(Tset, aPairEst[5], yerr = np.std(aPairEst[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: min')
# plt.errorbar(Tset, bPairEst[5], yerr = np.std(bPairEst[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: max')
# plt.errorbar(Tset, aPairEst[6], yerr = np.std(aPairEst[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: min')
# plt.errorbar(Tset, bPairEst[6], yerr = np.std(bPairEst[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: max')
# plt.errorbar(Tset, aPairEst[7], yerr = np.std(aPairEst[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: min')
# plt.errorbar(Tset, bPairEst[7], yerr = np.std(bPairEst[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (min/max pair)")
plt.title("How T affects error of unbiased estimator (min/max pair)")
plt.savefig("Femnist_t_mid_est_min_max.png")
plt.clf()

# plot ranking preservations for each T
plt.errorbar(Tset, aPercSmall[0], yerr = np.std(aPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(Tset, aPercLarge[0], yerr = np.std(aPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(Tset, aPercSmall[1], yerr = np.std(aPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(Tset, aPercLarge[1], yerr = np.std(aPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(Tset, aPercSmall[2], yerr = np.std(aPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(Tset, aPercLarge[2], yerr = np.std(aPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(Tset, aPercSmall[3], yerr = np.std(aPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(Tset, aPercLarge[3], yerr = np.std(aPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
# plt.errorbar(Tset, aPercSmall[4], yerr = np.std(aPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
# plt.errorbar(Tset, aPercLarge[4], yerr = np.std(aPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(Tset, aPercSmall[5], yerr = np.std(aPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(Tset, aPercLarge[5], yerr = np.std(aPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(Tset, aPercSmall[6], yerr = np.std(aPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(Tset, aPercLarge[6], yerr = np.std(aPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(Tset, aPercSmall[7], yerr = np.std(aPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(Tset, aPercLarge[7], yerr = np.std(aPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for true distribution")
plt.savefig("Femnist_t_mid_perc_ratio.png")
plt.clf()

plt.errorbar(Tset, bPercSmall[0], yerr = np.std(bPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(Tset, bPercLarge[0], yerr = np.std(bPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(Tset, bPercSmall[1], yerr = np.std(bPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(Tset, bPercLarge[1], yerr = np.std(bPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(Tset, bPercSmall[2], yerr = np.std(bPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(Tset, bPercLarge[2], yerr = np.std(bPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(Tset, bPercSmall[3], yerr = np.std(bPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(Tset, bPercLarge[3], yerr = np.std(bPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
# plt.errorbar(Tset, bPercSmall[4], yerr = np.std(bPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
# plt.errorbar(Tset, bPercLarge[4], yerr = np.std(bPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(Tset, bPercSmall[5], yerr = np.std(bPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(Tset, bPercLarge[5], yerr = np.std(bPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(Tset, bPercSmall[6], yerr = np.std(bPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(Tset, bPercLarge[6], yerr = np.std(bPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(Tset, bPercSmall[7], yerr = np.std(bPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(Tset, bPercLarge[7], yerr = np.std(bPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (sum)")
plt.savefig("Femnist_t_mid_perc_sum.png")
plt.clf()

plt.errorbar(Tset, cPercSmall[0], yerr = np.std(cPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(Tset, cPercLarge[0], yerr = np.std(cPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(Tset, cPercSmall[1], yerr = np.std(cPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(Tset, cPercLarge[1], yerr = np.std(cPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(Tset, cPercSmall[2], yerr = np.std(cPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(Tset, cPercLarge[2], yerr = np.std(cPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(Tset, cPercSmall[3], yerr = np.std(cPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(Tset, cPercLarge[3], yerr = np.std(cPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
# plt.errorbar(Tset, cPercSmall[4], yerr = np.std(cPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
# plt.errorbar(Tset, cPercLarge[4], yerr = np.std(cPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(Tset, cPercSmall[5], yerr = np.std(cPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(Tset, cPercLarge[5], yerr = np.std(cPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(Tset, cPercSmall[6], yerr = np.std(cPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(Tset, cPercLarge[6], yerr = np.std(cPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(Tset, cPercSmall[7], yerr = np.std(cPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(Tset, cPercLarge[7], yerr = np.std(cPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (min pair)")
plt.savefig("Femnist_t_mid_perc_min.png")
plt.clf()

plt.errorbar(Tset, dPercSmall[0], yerr = np.std(dPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(Tset, dPercLarge[0], yerr = np.std(dPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(Tset, dPercSmall[1], yerr = np.std(dPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(Tset, dPercLarge[1], yerr = np.std(dPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(Tset, dPercSmall[2], yerr = np.std(dPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(Tset, dPercLarge[2], yerr = np.std(dPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(Tset, dPercSmall[3], yerr = np.std(dPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(Tset, dPercLarge[3], yerr = np.std(dPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
# plt.errorbar(Tset, dPercSmall[4], yerr = np.std(dPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
# plt.errorbar(Tset, dPercLarge[4], yerr = np.std(dPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(Tset, dPercSmall[5], yerr = np.std(dPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(Tset, dPercLarge[5], yerr = np.std(dPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(Tset, dPercSmall[6], yerr = np.std(dPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(Tset, dPercLarge[6], yerr = np.std(dPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(Tset, dPercSmall[7], yerr = np.std(dPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(Tset, dPercLarge[7], yerr = np.std(dPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (max pair)")
plt.savefig("Femnist_t_mid_perc_max.png")

# compute total runtime in minutes and seconds
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
