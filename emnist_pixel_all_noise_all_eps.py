"""Modules provide various time-related functions, generate pseudo-random numbers,
compute the natural logarithm of a number, remember the order in which items are added,
have cool visual feedback of the current throughput, create static, animated,
and interactive visualisations, provide functionality to automatically download
and cache the EMNIST dataset, compute the mean of a list quickly and accurately,
work with arrays, and carry out fast numerical computations in Python."""
import time
import random
from math import log
from collections import OrderedDict
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
from statistics import fmean
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
np.set_printoptions(suppress = True)
np.seterr(divide = 'ignore', invalid = 'ignore')
tf.random.set_seed(638)

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter()
print("\nStarting...")
random.seed(3249583)
np.random.seed(107642)

# LOAD TRAINING AND TEST SAMPLES FOR 'DIGITS' SUBSET
images1, labels1 = extract_training_samples('digits')
images2, labels2 = extract_test_samples('digits')

# COMBINE TRAINING AND TEST SAMPLES INTO ONE NP ARRAY
images = np.concatenate((images1, images2))
labels = np.concatenate((labels1, labels2))
print("Loading training and test samples...")

# NUMPY ARRAYS TO STORE LABELS ASSOCIATED WITH WHICH DIGIT
digitSet = np.zeros((10, 28000), dtype = int)
digitIndexSet = np.zeros((10, 28000), dtype = int)

# REPORT FREQUENCY OF EACH DIGIT (AND TOTAL)
digitFreq = np.zeros(10, dtype = int)
TOTAL_FREQ = 0

def add_digit(dg, im, imset, ixset, freq, tc):
    """Method adds digit to set, index to index set and increments freq."""
    imset[dg, freq[dg]] = im
    ixset[dg, freq[dg]] = tc
    freq[dg] = freq[dg] + 1

# SPLIT NUMBERS 0-9
for digit in labels:

    # CALL FUNCTION DEFINED ABOVE
    add_digit(digit, digit, digitSet, digitIndexSet, digitFreq, TOTAL_FREQ)
    TOTAL_FREQ = TOTAL_FREQ + 1

print("Splitting numbers 0-9...")

# SIMILAR ARRAYS TO STORE CONDENSED IMAGES ASSOCIATED WITH EACH DIGIT
smallPic = np.zeros((4, 4))
digitImSet = np.zeros((10, 28000, 4, 4))
digitImIxSet = np.zeros((10, 28000), dtype = int)

# REPORT FREQUENCY OF EACH IMAGE (AND TOTAL)
digitImFreq = np.zeros(10, dtype = int)
IMAGE_FREQ = 0

print("\nPreprocessing images...")

with alive_bar(len(images)) as bar:
    for pic in images:

        # PARTITION EACH IMAGE INTO 16 7x7 SUBIMAGES
        for i in range(4):
            for j in range(4):
                subImage = pic[7*i : 7*(i + 1), 7*j : 7*(j + 1)]

                # SAVE ROUNDED MEAN OF EACH SUBIMAGE INTO CORRESPONDING CELL OF SMALLPIC
                meanSubImage = np.mean(subImage)
                if meanSubImage >= 128:
                    smallPic[i, j] = 1
                else:
                    smallPic[i, j] = 0

        # SPLIT IMAGES BY ASSOCIATION WITH PARTICULAR LABEL
        for digit in range(0, 10):
            if IMAGE_FREQ in digitIndexSet[digit]:
                add_digit(digit, smallPic, digitImSet, digitImIxSet, digitImFreq, IMAGE_FREQ)
                break

        IMAGE_FREQ = IMAGE_FREQ + 1
        bar()

# NUMBER OF SAMPLES COVERS APPROX 5% OF IMAGES
T = 1400

# STORE T IMAGES CORRESPONDING TO EACH DIGIT
sampleImSet = np.zeros((10, T, 4, 4))
sampleImList = np.zeros((10*T, 4, 4))
sizeUniqueImSet = np.zeros(10)
OVERALL_FREQ = 0

print("\nFinding unique images...")

for D in range(0, 10):

    # RANDOMLY SAMPLE T INDICES FROM EACH DIGIT SET
    randomIndices = random.sample(range(0, 28000), T)
    SAMPLE_FREQ = 0

    for index in randomIndices:

        # EXTRACT EACH IMAGE CORRESPONDING TO EACH OF THE T INDICES AND SAVE IN NEW STRUCTURE
        randomImage = digitImSet[D, index]
        sampleImSet[D, SAMPLE_FREQ] = randomImage
        sampleImList[OVERALL_FREQ] = randomImage
        SAMPLE_FREQ = SAMPLE_FREQ + 1
        OVERALL_FREQ = OVERALL_FREQ + 1

    # FIND FREQUENCIES OF ALL UNIQUE IMAGES IN SAMPLE IMAGE SET
    uniqueImSet = np.unique(sampleImSet[D], axis = 0)
    sizeUniqueImSet[D] = len(uniqueImSet)

# FIND FREQUENCIES OF UNIQUE IMAGES IN SAMPLE IMAGE LIST
uniqueImList = np.unique(sampleImList, axis = 0)
sizeUniqueImList = len(uniqueImList)

# DOMAIN FOR EACH DIGIT DISTRIBUTION IS NUMBER OF UNIQUE IMAGES
U = 207

# STORE FREQUENCIES OF UNIQUE IMAGES FOR EACH DIGIT
uImageSet = np.zeros((10, U, 4, 4))
uFreqSet = np.zeros((10, U))
uProbsSet = np.zeros((10, U))

print("Creating probability distributions...")

# SMOOTHING PARAMETER: 0.1 AND 1 ARE TOO LARGE
ALPHA = 0.01

for D in range(0, 10):
    UNIQUE_FREQ = 0

    # STORE IMAGE AND SMOOTHED PROBABILITY AS WELL AS FREQUENCY
    for image in uniqueImList:
        where = np.where(np.all(image == sampleImSet[D], axis = (1, 2)))
        freq = len(where[0])
        uImageSet[D, UNIQUE_FREQ] = image
        uFreqSet[D, UNIQUE_FREQ] = int(freq)
        uProbsSet[D, UNIQUE_FREQ] = float((freq + ALPHA)/(T + (ALPHA*(sizeUniqueImSet[D]))))
        UNIQUE_FREQ = UNIQUE_FREQ + 1

# FOR K3 ESTIMATOR (SCHULMAN) TAKE A SMALL SAMPLE OF UNIQUE IMAGES
E = 10

# STORE IMAGES, FREQUENCIES AND PROBABILITIES FOR THIS SUBSET
eImageSet = np.ones((10, E, 4, 4))
eFreqSet = np.zeros((10, E))
eProbsSet = np.zeros((10, E))
eTotalFreq = np.zeros(10)

uSampledSet = np.random.choice(U, E, replace = False)

# BORROW DATA FROM CORRESPONDING INDICES OF MAIN IMAGE AND FREQUENCY SETS
for D in range(0, 10):
    for i in range(E):
        eImageSet[D, i] = uImageSet[D, uSampledSet[i]]
        eFreqSet[D, i] = uFreqSet[D, uSampledSet[i]]
        eTotalFreq[D] = sum(eFreqSet[D])
        eProbsSet[D, i] = float((eFreqSet[D, i] + ALPHA)/(T + (ALPHA*(eTotalFreq[D]))))

# PARAMETERS FOR THE ADDITION OF LAPLACE AND GAUSSIAN NOISE
DTA = 0.1
A = 0
R = 10

# LISTS OF THE VALUES OF EPSILON AND TRIALS THAT WILL BE RUN
epsset = [0.001, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]
trialset = ["mid_lap", "mid_lap_mc", "mid_gauss", "mid_gauss_mc", "end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc"]
ES = len(epsset)
TS = len(trialset)

# STORES FOR SUM, MIN/MAX DISTRIBUTIONS AND LAMBDA
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

# STORES FOR RANKING PRESERVATION ANALYSIS
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
for trial in range(5):

    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    INDEX_FREQ = 0

    for eps in epsset:
        for rep in range(R):

            print(f"\nTrial {trial + 1}: epsilon = {eps}, repeat = {rep + 1}...")

            # STORES FOR EXACT AND NOISY UNKNOWN DISTRIBUTIONS
            uDist = np.zeros((10, 10, U))
            nDist = np.zeros((10, 10, E, R))
            uList = []
            uCDList = []

            # STORES FOR RATIO BETWEEN UNKNOWN AND KNOWN DISTRIBUTIONS
            rList = []
            kList = []
            kCDList = []

            # STORES FOR BINARY SEARCH
            zeroUList = []
            zeroCDList = []
            oneUList = []
            oneCDList = []
            halfUList = []
            halfCDList = []

            # OPTION 1A: BASELINE CASE
            if trial % 2 == 0:
                b1 = log(2) / eps

            # OPTION 1B: MONTE CARLO ESTIMATE
            else:
                b1 = (1 + log(2)) / eps

            b2 = (2*((log(1.25))/DTA)*b1) / eps

            # OPTION 2A: ADD LAPLACE NOISE
            if trial % 4 == 0 or trial % 4 == 1:
                noiseLG = tfp.distributions.Laplace(loc = A, scale = b1)
        
            # OPTION 2B: ADD GAUSSIAN NOISE
            else:
                noiseLG = tfp.distributions.Normal(loc = A, scale = b2)

            def unbias_est(lda, rlist, klist, ulist, cdlist):
                """Compute sum of unbiased estimators corresponding to all pairs."""
                count = 1

                for row in range(0, len(rlist)):
                    uest = ((lda * (rlist[row] - 1)) - log(rlist[row])) / T
          
                    # OPTION 3B: ADD NOISE AT END
                    if trial >= 4:
                        err = noiseLG.sample(sample_shape = (1,)).numpy()[0]
                    else:
                        err = 0.0

                    # ADD NOISE TO UNKNOWN DISTRIBUTION ESTIMATOR THEN COMPARE TO KNOWN DISTRIBUTION
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
                    uest = ((lda * (rlist[row] - 1)) - log(rlist[row])) / T

                    # OPTION 3B: ADD NOISE AT END
                    if trial >= 4:
                        err = noiseLG.sample(sample_shape = (1,)).numpy()[0]
                    else:
                        err = 0.0

                    # ADD NOISE TO UNKNOWN DISTRIBUTION ESTIMATOR THEN COMPARE TO KNOWN DISTRIBUTION
                    err = abs(err + uest - klist[row])

                    if err != 0.0:
                        ulist.append(err)

                        c = count // 10
                        d = count % 10

                        ulist.append((c, d))

                        if c == d + 1:
                            count = count + 2
                        else:
                            count = count + 1

                mi = cdlist.index(mp)
                return ulist[mi]

            # FOR EACH COMPARISON DIGIT COMPUTE EXACT AND NOISY UNKNOWN DISTRIBUTIONS FOR ALL DIGITS
            for C in range(0, 10):
                for D in range(0, 10):

                    for i in range(0, U):
                        uDist[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

                    for j in range(0, E):
                        nDist[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

                        # OPTION 3A: ADD NOISE IN MIDDLE
                        if trial < 4:
                            nDist[C, D, j] = nDist[C, D, j] + noiseLG.sample(sample_shape = (1,))

                    # ELIMINATE ALL ZERO VALUES WHEN DIGITS ARE IDENTICAL
                    if sum(uDist[C, D]) != 0.0:
                        uList.append(sum(uDist[C, D]))
                        uCDList.append((C, D))

                    # COMPUTE RATIO BETWEEN EXACT AND NOISY UNKNOWN DISTRIBUTIONS
                    ratio = abs(sum(nDist[C, D, j]) / sum(uDist[C, D]))

                    # ELIMINATE ALL DIVIDE BY ZERO ERRORS
                    if ratio != 0.0 and sum(uDist[C, D]) != 0.0:
                        rList.append(ratio)

                        # COMPUTE KNOWN DISTRIBUTION
                        kDist = abs(sum(nDist[C, D, j]) * log(ratio))
                        kList.append(kDist)
                        kCDList.append((C, D))

                        # WAIT UNTIL FINAL DIGIT PAIR (9, 8) TO ANALYSE EXACT UNKNOWN DISTRIBUTION LIST
                        if C == 9 and D == 8:

                            low = 0
                            high = 1
                            mid = 0.5

                            # COMPUTE UNBIASED ESTIMATORS WITH LAMBDA 0, 1, 0.5 THEN BINARY SEARCH
                            lowSum = unbias_est(low, rList, kList, zeroUList, zeroCDList)
                            highSum = unbias_est(high, rList, kList, oneUList, oneCDList)
                            minSum[trial, INDEX_FREQ, rep] = unbias_est(mid, rList, kList, halfUList, halfCDList)

                            # TOLERANCE BETWEEN BINARY SEARCH LIMITS ALWAYS GETS SMALL ENOUGH
                            while abs(high - low) > 0.00000001:

                                lowUList = []
                                lowCDList = []
                                highUList = []
                                highCDList = []
                                sumUList = []
                                sumCDList = []

                                lowSum = unbias_est(low, rList, kList, lowUList, lowCDList)
                                highSum = unbias_est(high, rList, kList, highUList, highCDList)

                                # REDUCE / INCREASE BINARY SEARCH LIMIT DEPENDING ON ABSOLUTE VALUE
                                if abs(lowSum) < abs(highSum):
                                    high = mid
                                else:
                                    low = mid

                                # SET NEW MIDPOINT
                                mid = (0.5*abs((high - low))) + low
                                minSum[trial, INDEX_FREQ, rep] = unbias_est(mid, rList, kList, sumUList, sumCDList)

                            sumLambda[trial, INDEX_FREQ, rep] = mid

                            # EXTRACT MIN PAIR BY ABSOLUTE VALUE OF EXACT UNKNOWN DISTRIBUTION
                            absUList = [abs(ul) for ul in uList]
                            minUList = sorted(absUList)
                            minAbs = minUList[0]
                            minIndex = uList.index(minAbs)
                            minPair = uCDList[minIndex]
                            MIN_COUNT = 1

                            # IF MIN PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT SMALLEST
                            while minPair not in halfCDList:        
                                minAbs = minUList[MIN_COUNT]
                                minIndex = uList.index(minAbs)
                                minPair = uCDList[minIndex]
                                MIN_COUNT = MIN_COUNT + 1

                            midMinIndex = halfCDList.index(minPair)
                            minPairEst[trial, INDEX_FREQ, rep] = halfUList[midMinIndex]

                            low = 0
                            high = 1
                            mid = 0.5

                            # FIND OPTIMAL LAMBDA FOR MIN PAIR
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

                            # EXTRACT MAX PAIR BY REVERSING UNKNOWN DISTRIBUTION LIST
                            maxUList = sorted(absUList, reverse = True)
                            maxAbs = maxUList[0]
                            maxIndex = uList.index(maxAbs)
                            maxPair = uCDList[maxIndex]
                            MAX_COUNT = 1

                            # IF MAX PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT LARGEST
                            while maxPair not in halfCDList:        
                                maxAbs = maxUList[MAX_COUNT]
                                maxIndex = uList.index(maxAbs)
                                maxPair = uCDList[maxIndex]
                                MAX_COUNT = MAX_COUNT + 1

                            midMaxIndex = halfCDList.index(maxPair)
                            maxPairEst[trial, INDEX_FREQ, rep] = halfUList[midMaxIndex]

                            low = 0
                            high = 1
                            mid = 0.5

                            # FIND OPTIMAL LAMBDA FOR MAX PAIR
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
            
            # UNKNOWN DISTRIBUTION IS IDENTICAL FOR ALL TRIALS, EPSILONS AND REPEATS
            if trial == 0 and eps == 0.001 and rep == 0:
                orderfile = open("emnist_unknown_dist_in_order.txt", "w", encoding = 'utf-8')
                orderfile.write("EMNIST: Unknown Distribution In Order\n")
                orderfile.write("Smaller corresponds to more similar digits\n\n")

                for i in oUDict:
                    orderfile.write(f"{i} : {oUDict[i]}\n")

            # COMPUTE RANKING PRESERVATION STATISTICS FOR EACH REPEAT
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
        
        # SUM UP REPEATS FOR ALL THE MAIN STATISTICS
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

        statsfile = open(f"emnist_{trialset[trial]}_noise_eps_{eps}.txt", "w", encoding = 'utf-8')
        statsfile.write(f"EMNIST: Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
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

# PLOT LAMBDAS FOR EACH EPSILON
print(f"\naLambda: {aLambda[0]}")
plt.errorbar(epsset, aLambda[0], yerr = np.std(aLambda[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap')
plt.errorbar(epsset, aLambda[1], yerr = np.std(aLambda[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc')
plt.errorbar(epsset, aLambda[2], yerr = np.std(aLambda[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss')
plt.errorbar(epsset, aLambda[3], yerr = np.std(aLambda[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc')
plt.errorbar(epsset, aLambda[4], yerr = np.std(aLambda[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap')
# plt.errorbar(epsset, aLambda[5], yerr = np.std(aLambda[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc')
# plt.errorbar(epsset, aLambda[6], yerr = np.std(aLambda[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss')
# plt.errorbar(epsset, aLambda[7], yerr = np.std(aLambda[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How epsilon affects lambda (sum)")
plt.savefig("Emnist_eps_mid_lambda_sum.png")
plt.clf()

plt.errorbar(epsset, aPairLambda[0], yerr = np.std(aPairLambda[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: min')
plt.errorbar(epsset, bPairLambda[0], yerr = np.std(aPairLambda[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: max')
plt.errorbar(epsset, aPairLambda[1], yerr = np.std(aPairLambda[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: min')
plt.errorbar(epsset, bPairLambda[1], yerr = np.std(aPairLambda[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: max')
plt.errorbar(epsset, aPairLambda[2], yerr = np.std(aPairLambda[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: min')
plt.errorbar(epsset, bPairLambda[2], yerr = np.std(aPairLambda[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: max')
plt.errorbar(epsset, aPairLambda[3], yerr = np.std(aPairLambda[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: min')
plt.errorbar(epsset, bPairLambda[3], yerr = np.std(aPairLambda[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: max')
plt.errorbar(epsset, aPairLambda[4], yerr = np.std(aPairLambda[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: min')
plt.errorbar(epsset, bPairLambda[4], yerr = np.std(aPairLambda[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: max')
# plt.errorbar(epsset, aPairLambda[5], yerr = np.std(aPairLambda[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: min')
# plt.errorbar(epsset, bPairLambda[5], yerr = np.std(aPairLambda[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: max')
# plt.errorbar(epsset, aPairLambda[6], yerr = np.std(aPairLambda[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: min')
# plt.errorbar(epsset, bPairLambda[6], yerr = np.std(aPairLambda[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: max')
# plt.errorbar(epsset, aPairLambda[7], yerr = np.std(aPairLambda[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: min')
# plt.errorbar(epsset, bPairLambda[7], yerr = np.std(aPairLambda[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How epsilon affects lambda (min/max pair)")
plt.savefig("Emnist_eps_mid_lambda_min_max.png")
plt.clf()

# PLOT SUM / ESTIMATES FOR EACH EPSILON
print(f"aSum: {aSum[0]}")
plt.errorbar(epsset, aSum[0], yerr = np.std(aSum[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap')
plt.errorbar(epsset, aSum[1], yerr = np.std(aSum[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc')
plt.errorbar(epsset, aSum[2], yerr = np.std(aSum[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss')
plt.errorbar(epsset, aSum[3], yerr = np.std(aSum[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc')
plt.errorbar(epsset, aSum[4], yerr = np.std(aSum[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap')
# plt.errorbar(epsset, aSum[5], yerr = np.std(aSum[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc')
# plt.errorbar(epsset, aSum[6], yerr = np.std(aSum[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss')
# plt.errorbar(epsset, aSum[7], yerr = np.std(aSum[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (sum)")
plt.title("How epsilon affects error of unbiased estimator (sum)")
plt.savefig("Emnist_eps_mid_est_sum.png")
plt.clf()

plt.errorbar(epsset, aPairEst[0], yerr = np.std(aPairEst[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: min')
plt.errorbar(epsset, bPairEst[0], yerr = np.std(aPairEst[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: max')
plt.errorbar(epsset, aPairEst[1], yerr = np.std(aPairEst[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: min')
plt.errorbar(epsset, bPairEst[1], yerr = np.std(aPairEst[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: max')
plt.errorbar(epsset, aPairEst[2], yerr = np.std(aPairEst[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: min')
plt.errorbar(epsset, bPairEst[2], yerr = np.std(aPairEst[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: max')
plt.errorbar(epsset, aPairEst[3], yerr = np.std(aPairEst[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: min')
plt.errorbar(epsset, bPairEst[3], yerr = np.std(aPairEst[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: max')
plt.errorbar(epsset, aPairEst[4], yerr = np.std(aPairEst[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: min')
plt.errorbar(epsset, bPairEst[4], yerr = np.std(aPairEst[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: max')
# plt.errorbar(epsset, aPairEst[5], yerr = np.std(aPairEst[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: min')
# plt.errorbar(epsset, bPairEst[5], yerr = np.std(aPairEst[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: max')
# plt.errorbar(epsset, aPairEst[6], yerr = np.std(aPairEst[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: min')
# plt.errorbar(epsset, bPairEst[6], yerr = np.std(aPairEst[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: max')
# plt.errorbar(epsset, aPairEst[7], yerr = np.std(aPairEst[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: min')
# plt.errorbar(epsset, bPairEst[7], yerr = np.std(aPairEst[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (min/max pair)")
plt.title("How epsilon affects error of unbiased estimator (min/max pair)")
plt.savefig("Emnist_eps_mid_est_min_max.png")
plt.clf()

# PLOT RANKING PRESERVATIONS FOR EACH EPSILON
plt.errorbar(epsset, aPercSmall[0], yerr = np.std(aPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(epsset, aPercLarge[0], yerr = np.std(aPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(epsset, aPercSmall[1], yerr = np.std(aPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(epsset, aPercLarge[1], yerr = np.std(aPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(epsset, aPercSmall[2], yerr = np.std(aPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(epsset, aPercLarge[2], yerr = np.std(aPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(epsset, aPercSmall[3], yerr = np.std(aPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(epsset, aPercLarge[3], yerr = np.std(aPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
plt.errorbar(epsset, aPercSmall[4], yerr = np.std(aPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
plt.errorbar(epsset, aPercLarge[4], yerr = np.std(aPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(epsset, aPercSmall[5], yerr = np.std(aPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(epsset, aPercLarge[5], yerr = np.std(aPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(epsset, aPercSmall[6], yerr = np.std(aPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(epsset, aPercLarge[6], yerr = np.std(aPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(epsset, aPercSmall[7], yerr = np.std(aPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(epsset, aPercLarge[7], yerr = np.std(aPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for true distribution")
plt.savefig("Emnist_eps_mid_perc_ratio.png")
plt.clf()

plt.errorbar(epsset, bPercSmall[0], yerr = np.std(bPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(epsset, bPercLarge[0], yerr = np.std(bPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(epsset, bPercSmall[1], yerr = np.std(bPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(epsset, bPercLarge[1], yerr = np.std(bPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(epsset, bPercSmall[2], yerr = np.std(bPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(epsset, bPercLarge[2], yerr = np.std(bPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(epsset, bPercSmall[3], yerr = np.std(bPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(epsset, bPercLarge[3], yerr = np.std(bPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
plt.errorbar(epsset, bPercSmall[4], yerr = np.std(bPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
plt.errorbar(epsset, bPercLarge[4], yerr = np.std(bPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(epsset, bPercSmall[5], yerr = np.std(bPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(epsset, bPercLarge[5], yerr = np.std(bPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(epsset, bPercSmall[6], yerr = np.std(bPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(epsset, bPercLarge[6], yerr = np.std(bPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(epsset, bPercSmall[7], yerr = np.std(bPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(epsset, bPercLarge[7], yerr = np.std(bPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (sum)")
plt.savefig("Emnist_eps_mid_perc_sum.png")
plt.clf()

plt.errorbar(epsset, cPercSmall[0], yerr = np.std(cPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(epsset, cPercLarge[0], yerr = np.std(cPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(epsset, cPercSmall[1], yerr = np.std(cPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(epsset, cPercLarge[1], yerr = np.std(cPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(epsset, cPercSmall[2], yerr = np.std(cPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(epsset, cPercLarge[2], yerr = np.std(cPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(epsset, cPercSmall[3], yerr = np.std(cPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(epsset, cPercLarge[3], yerr = np.std(cPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
plt.errorbar(epsset, cPercSmall[4], yerr = np.std(cPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
plt.errorbar(epsset, cPercLarge[4], yerr = np.std(cPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(epsset, cPercSmall[5], yerr = np.std(cPercSmall[5], axis = 0),  = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(epsset, cPercLarge[5], yerr = np.std(cPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(epsset, cPercSmall[6], yerr = np.std(cPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(epsset, cPercLarge[6], yerr = np.std(cPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(epsset, cPercSmall[7], yerr = np.std(cPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(epsset, cPercLarge[7], yerr = np.std(cPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (min pair)")
plt.savefig("Emnist_eps_mid_perc_min.png")
plt.clf()

plt.errorbar(epsset, dPercSmall[0], yerr = np.std(dPercSmall[0], axis = 0), color = 'tab:brown', marker = 'o', label = 'mid lap: smallest 10%')
plt.errorbar(epsset, dPercLarge[0], yerr = np.std(dPercLarge[0], axis = 0), color = 'tab:brown', marker = 'x', label = 'mid lap: largest 10%')
plt.errorbar(epsset, dPercSmall[1], yerr = np.std(dPercSmall[1], axis = 0), color = 'tab:purple', marker = 'o', label = 'mid lap mc: smallest 10%')
plt.errorbar(epsset, dPercLarge[1], yerr = np.std(dPercLarge[1], axis = 0), color = 'tab:purple', marker = 'x', label = 'mid lap mc: largest 10%')
plt.errorbar(epsset, dPercSmall[2], yerr = np.std(dPercSmall[2], axis = 0), color = 'tab:blue', marker = 'o', label = 'mid gauss: smallest 10%')
plt.errorbar(epsset, dPercLarge[2], yerr = np.std(dPercLarge[2], axis = 0), color = 'tab:blue', marker = 'x', label = 'mid gauss: largest 10%')
plt.errorbar(epsset, dPercSmall[3], yerr = np.std(dPercSmall[3], axis = 0), color = 'tab:cyan', marker = 'o', label = 'mid gauss mc: smallest 10%')
plt.errorbar(epsset, dPercLarge[3], yerr = np.std(dPercLarge[3], axis = 0), color = 'tab:cyan', marker = 'x', label = 'mid gauss mc: largest 10%')
plt.errorbar(epsset, dPercSmall[4], yerr = np.std(dPercSmall[4], axis = 0), color = 'tab:olive', marker = 'o', label = 'end lap: smallest 10%')
plt.errorbar(epsset, dPercLarge[4], yerr = np.std(dPercLarge[4], axis = 0), color = 'tab:olive', marker = 'x', label = 'end lap: largest 10%')
# plt.errorbar(epsset, dPercSmall[5], yerr = np.std(dPercSmall[5], axis = 0), color = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(epsset, dPercLarge[5], yerr = np.std(dPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(epsset, dPercSmall[6], yerr = np.std(dPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(epsset, dPercLarge[6], yerr = np.std(dPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(epsset, dPercSmall[7], yerr = np.std(dPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(epsset, dPercLarge[7], yerr = np.std(dPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (max pair)")
plt.savefig("Emnist_eps_mid_perc_max.png")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
