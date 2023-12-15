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

# KEEP TRACK OF HOW MANY OF EACH DIGIT (AND TOTAL) ARE PROCESSED
digitCount = np.zeros(10, dtype = int)
TOTAL_COUNT = 0

def add_digit(dg, im, imset, ixset, count, tc):
    """Method adds digit to set, index to index set and increments count."""
    imset[dg, count[dg]] = im
    ixset[dg, count[dg]] = tc
    count[dg] = count[dg] + 1

# SPLIT NUMBERS 0-9
for digit in labels:

    # CALL FUNCTION DEFINED ABOVE
    add_digit(digit, digit, digitSet, digitIndexSet, digitCount, TOTAL_COUNT)
    TOTAL_COUNT = TOTAL_COUNT + 1

print("Splitting numbers 0-9...")

# SIMILAR ARRAYS TO STORE CONDENSED IMAGES ASSOCIATED WITH EACH DIGIT
smallPic = np.zeros((4, 4))
digitImSet = np.zeros((10, 28000, 4, 4))
digitImIxSet = np.zeros((10, 28000), dtype = int)

# KEEP TRACK OF HOW MANY OF EACH IMAGE (AND TOTAL) ARE PROCESSED
digitImCount = np.zeros(10, dtype = int)
IMAGE_COUNT = 0

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
            if IMAGE_COUNT in digitIndexSet[digit]:
                add_digit(digit, smallPic, digitImSet, digitImIxSet, digitImCount, IMAGE_COUNT)
                break

        IMAGE_COUNT = IMAGE_COUNT + 1
        bar()

# INVESTIGATE SAMPLES FROM APPROX 1% TO APPROX 20% OF IMAGES
Tset = [280, 560, 840, 1120, 1400, 1750, 2100, 2800, 3500, 4200, 4900, 5600]
ES = len(Tset)
INDEX_COUNT = 0

for T in Tset:

    # STORE T IMAGES CORRESPONDING TO EACH DIGIT
    sampleImSet = np.zeros((10, T, 4, 4))
    sampleImList = np.zeros((10*T, 4, 4))
    sizeUniqueImSet = np.zeros(10)
    OVERALL_COUNT = 0

    print("\nFinding unique images...")

    for D in range(0, 10):

        # RANDOMLY SAMPLE T INDICES FROM EACH DIGIT SET
        randomIndices = random.sample(range(0, 28000), T)
        SAMPLE_COUNT = 0

        for index in randomIndices:

            # EXTRACT EACH IMAGE CORRESPONDING TO EACH OF THE T INDICES AND SAVE IN NEW STRUCTURE
            randomImage = digitImSet[D, index]
            sampleImSet[D, SAMPLE_COUNT] = randomImage
            sampleImList[OVERALL_COUNT] = randomImage
            SAMPLE_COUNT = SAMPLE_COUNT + 1
            OVERALL_COUNT = OVERALL_COUNT + 1

        # FIND COUNTS OF ALL UNIQUE IMAGES IN SAMPLE IMAGE SET
        uniqueImSet = np.unique(sampleImSet[D], axis = 0)
        sizeUniqueImSet[D] = len(uniqueImSet)

    # FIND COUNTS OF UNIQUE IMAGES IN SAMPLE IMAGE LIST
    uniqueImList = np.unique(sampleImList, axis = 0)

    # DOMAIN FOR EACH DIGIT DISTRIBUTION IS NUMBER OF UNIQUE IMAGES
    U = len(uniqueImList)

    # FIND AND STORE FREQUENCIES OF UNIQUE IMAGES FOR EACH DIGIT
    uImageSet = np.zeros((10, U, 4, 4))
    uFreqSet = np.zeros((10, U))
    uProbsSet = np.zeros((10, U))

    print("Creating probability distributions...")

    # SMOOTHING PARAMETER: 0.1 AND 1 ARE TOO LARGE
    ALPHA = 0.01

    for D in range(0, 10):
        UNIQUE_COUNT = 0

        # STORE IMAGE AND SMOOTHED PROBABILITY AS WELL AS FREQUENCY
        for image in uniqueImList:
            where = np.where(np.all(image == sampleImSet[D], axis = (1, 2)))
            freq = len(where[0])
            uImageSet[D, UNIQUE_COUNT] = image
            uFreqSet[D, UNIQUE_COUNT] = int(freq)
            uProbsSet[D, UNIQUE_COUNT] = float((freq + ALPHA)/(T + (ALPHA*(sizeUniqueImSet[D]))))
            UNIQUE_COUNT = UNIQUE_COUNT + 1

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
    EPS = 0.1
    DTA = 0.1
    A = 0
    R = 10

    # LIST OF THE TRIALS THAT WILL BE RUN
    trialset = ["mid_lap", "mid_lap_mc", "mid_gauss", "mid_gauss_mc", "end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc"]
    TS = len(trialset)

    # STORES FOR SUM, MIN/MAX KLD AND LAMBDA
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
    tPercSmall = np.zeros((TS, ES, R))
    tPercLarge = np.zeros((TS, ES, R))
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

            # STORES FOR EXACT KLD
            KLDiv = np.zeros((10, 10, U))
            KList = []
            CDList = []

            # STORES FOR ESTIMATED KLD
            eKLDiv = np.zeros((10, 10, E, R))
            eKList = []

            # STORES FOR RATIO BETWEEN KLDS AND TRUE DISTRIBUTION
            rKList = []
            tKList = []
            tCDList = []

            # STORES FOR UNBIASED ESTIMATE OF KLD
            zeroKList = []
            zeroCDList = []
            oneKList = []
            oneCDList = []
            halfKList = []
            halfCDList = []

            # OPTION 1A: QUERYING ENTIRE DISTRIBUTION
            if trial % 2 == 0:
                b1 = log(2) / EPS

            # OPTION 1B: MONTE CARLO SAMPLING
            else:
                b1 = (1 + log(2)) / EPS

            b2 = (2*((log(1.25))/DTA)*b1) / EPS

            # OPTION 2A: ADD LAPLACE NOISE
            if trial % 4 == 0 or trial % 4 == 1:
                noiseLG = tfp.distributions.Laplace(loc = A, scale = b1)
        
            # OPTION 2B: ADD GAUSSIAN NOISE
            else:
                noiseLG = tfp.distributions.Normal(loc = A, scale = b2)

            def unbias_est(lda, rklist, tklist, lklist, lcdlist):
                """Compute sum of unbiased estimators corresponding to all pairs."""
                count = 1

                for row in range(0, len(rklist)):
                    lest = ((lda * (rklist[row] - 1)) - log(rklist[row])) / T
          
                    # OPTION 3B: ADD NOISE AT END
                    if trial >= 4:
                        err = noiseLG.sample(sample_shape = (1,)).numpy()[0]
                    else:
                        err = 0.0

                    # ADD NOISE TO UNBIASED ESTIMATOR THEN COMPARE TO TRUE DISTRIBUTION
                    err = abs(err + lest - tklist[row])

                    if err != 0.0:
                        lklist.append(err)

                        c = count // 10
                        d = count % 10

                        lcdlist.append((c, d))

                        if c == d + 1:
                            count = count + 2
                        else:
                            count = count + 1

                return sum(lklist)
    
            def min_max(lda, rklist, tklist, lklist, lcdlist, mp):
                """Compute unbiased estimator corresponding to min or max pair."""
                count = 1

                for row in range(0, len(rklist)):
                    lest = ((lda * (rklist[row] - 1)) - log(rklist[row])) / T

                    # OPTION 3B: ADD NOISE AT END
                    if trial >= 4:
                        err = noiseLG.sample(sample_shape = (1,)).numpy()[0]
                    else:
                        err = 0.0

                    # ADD NOISE TO UNBIASED ESTIMATOR THEN COMPARE TO TRUE DISTRIBUTION
                    err = abs(err + lest - tklist[row])

                    if err != 0.0:
                        lklist.append(err)

                        c = count // 10
                        d = count % 10

                        lcdlist.append((c, d))

                        if c == d + 1:
                            count = count + 2
                        else:
                            count = count + 1

                lmi = lcdlist.index(mp)
                return lklist[lmi]

            # FOR EACH COMPARISON DIGIT COMPUTE KLD FOR ALL DIGITS
            for C in range(0, 10):
                for D in range(0, 10):

                    for i in range(0, U):
                        KLDiv[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

                    for j in range(0, E):
                        eKLDiv[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

                        # OPTION 3A: ADD NOISE IN MIDDLE
                        if trial < 4:
                            eKLDiv[C, D, j] = eKLDiv[C, D, j] + noiseLG.sample(sample_shape = (1,))

                    # ELIMINATE ALL ZERO VALUES WHEN DIGITS ARE IDENTICAL
                    if sum(KLDiv[C, D]) != 0.0:
                        KList.append(sum(KLDiv[C, D]))
                        CDList.append((C, D))

                    # COMPUTE RATIO BETWEEN EXACT AND ESTIMATED KLD
                    ratio = abs(sum(eKLDiv[C, D, j]) / sum(KLDiv[C, D]))

                    # ELIMINATE ALL DIVIDE BY ZERO ERRORS
                    if ratio != 0.0 and sum(KLDiv[C, D]) != 0.0:
                        rKList.append(ratio)

                        # COMPUTE TRUE DISTRIBUTION
                        trueDist = abs(sum(eKLDiv[C, D, j]) * log(ratio))
                        tKList.append(trueDist)
                        tCDList.append((C, D))

                        # WAIT UNTIL FINAL DIGIT PAIR (9, 8) TO ANALYSE EXACT KLD LIST
                        if C == 9 and D == 8:

                            low = 0
                            high = 10
                            mid = 5

                            # COMPUTE UNBIASED ESTIMATORS WITH LAMBDA 0, 1, 0.5 THEN BINARY SEARCH
                            lowSum = unbias_est(low, rKList, tKList, zeroKList, zeroCDList)
                            highSum = unbias_est(high, rKList, tKList, oneKList, oneCDList)
                            minSum[trial, INDEX_COUNT, rep] = unbias_est(mid, rKList, tKList, halfKList, halfCDList)

                            # TOLERANCE BETWEEN BINARY SEARCH LIMITS ALWAYS GETS SMALL ENOUGH
                            while abs(high - low) > 0.00000001:

                                lowKList = []
                                lowCDList = []
                                highKList = []
                                highCDList = []
                                sumKList = []
                                sumCDList = []

                                lowSum = unbias_est(low, rKList, tKList, lowKList, lowCDList)
                                highSum = unbias_est(high, rKList, tKList, highKList, highCDList)

                                # REDUCE / INCREASE BINARY SEARCH LIMIT DEPENDING ON ABSOLUTE VALUE
                                if abs(lowSum) < abs(highSum):
                                    high = mid
                                else:
                                    low = mid

                                # SET NEW MIDPOINT
                                mid = (0.5*abs((high - low))) + low
                                minSum[trial, INDEX_COUNT, rep] = unbias_est(mid, rKList, tKList, sumKList, sumCDList)

                            sumLambda[trial, INDEX_COUNT, rep] = mid

                            # EXTRACT MIN PAIR BY ABSOLUTE VALUE OF EXACT KLD
                            absKList = [abs(kl) for kl in KList]
                            minKList = sorted(absKList)
                            minAbs = minKList[0]
                            minIndex = KList.index(minAbs)
                            minPair = CDList[minIndex]
                            MIN_COUNT = 1

                            # IF MIN PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT SMALLEST
                            while minPair not in halfCDList:        
                                minAbs = minKList[MIN_COUNT]
                                minIndex = KList.index(minAbs)
                                minPair = CDList[minIndex]
                                MIN_COUNT = MIN_COUNT + 1

                            midMinIndex = halfCDList.index(minPair)
                            minPairEst[trial, INDEX_COUNT, rep] = halfKList[midMinIndex]

                            low = 0
                            high = 10
                            mid = 5

                            # FIND OPTIMAL LAMBDA FOR MIN PAIR
                            while abs(high - low) > 0.00000001:

                                lowKList = []
                                lowCDList = []
                                highKList = []
                                highCDList = []
                                minKList = []
                                minCDList = []

                                lowMinKL = min_max(low, rKList, tKList, lowKList, lowCDList, minPair)
                                highMinKL = min_max(high, rKList, tKList, highKList, highCDList, minPair)

                                if abs(lowMinKL) < abs(highMinKL):
                                    high = mid
                                else:
                                    low = mid

                                mid = (0.5*abs((high - low))) + low
                                minPairEst[trial, INDEX_COUNT, rep] = min_max(mid, rKList, tKList, minKList, minCDList, minPair)

                            minPairLambda[trial, INDEX_COUNT, rep] = mid

                            # EXTRACT MAX PAIR BY REVERSING EXACT KLD LIST
                            maxKList = sorted(absKList, reverse = True)
                            maxAbs = maxKList[0]
                            maxIndex = KList.index(maxAbs)
                            maxPair = CDList[maxIndex]
                            MAX_COUNT = 1

                            # IF MAX PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT LARGEST
                            while maxPair not in halfCDList:        
                                maxAbs = maxKList[MAX_COUNT]
                                maxIndex = KList.index(maxAbs)
                                maxPair = CDList[maxIndex]
                                MAX_COUNT = MAX_COUNT + 1

                            midMaxIndex = halfCDList.index(maxPair)
                            maxPairEst[trial, INDEX_COUNT, rep] = halfKList[midMaxIndex]

                            low = 0
                            high = 10
                            mid = 5

                            # FIND OPTIMAL LAMBDA FOR MAX PAIR
                            while abs(high - low) > 0.00000001:

                                lowKList = []
                                lowCDList = []
                                highKList = []
                                highCDList = []
                                maxKList = []
                                maxCDList = []

                                lowMaxKL = min_max(low, rKList, tKList, lowKList, lowCDList, maxPair)
                                highMaxKL = min_max(high, rKList, tKList, highKList, highCDList, maxPair)
                
                                if abs(lowMaxKL) < abs(highMaxKL):
                                    high = mid
                                else:
                                    low = mid

                                mid = (0.5*(abs(high - low))) + low
                                maxPairEst[trial, INDEX_COUNT, rep] = min_max(mid, rKList, tKList, maxKList, maxCDList, maxPair)

                            maxPairLambda[trial, INDEX_COUNT, rep] = mid

            def rank_pres(bin, okld, tokld):
                """Do smallest/largest 10% in exact KLD remain in smaller/larger half of estimator?"""
                rows = 90
                num = 0

                if bin == 0: 
                    dict = list(okld.values())[0 : int(rows / 10)]
                    tdict = list(tokld.values())[0 : int(rows / 2)]
                else:
                    dict = list(okld.values())[int(9*(rows / 10)) : rows]
                    tdict = list(tokld.values())[int(rows / 2) : rows]

                for di in dict:
                    for dj in tdict:    
                        if dj == di:
                            num = num + 1

                return 100*(num / int(rows/10))

            KLDict = dict(zip(KList, CDList))
            orderedKLDict = OrderedDict(sorted(KLDict.items()))
            
            # EXACT KLD IS IDENTICAL FOR ALL TS, TRIALS AND REPEATS
            if T == 280 and trial == 0 and rep == 0:
                orderfile = open("emnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
                orderfile.write("EMNIST: Exact KL Divergence In Order\n")
                orderfile.write("Smaller corresponds to more similar digits\n\n")

                for i in orderedKLDict:
                    orderfile.write(f"{i} : {orderedKLDict[i]}\n")

            # COMPUTE RANKING PRESERVATION STATISTICS FOR EACH REPEAT
            tKLDict = dict(zip(tKList, tCDList))
            tOrderedKLDict = OrderedDict(sorted(tKLDict.items()))
            tPercSmall[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, tOrderedKLDict)
            tPercLarge[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, tOrderedKLDict)

            sumKLDict = dict(zip(sumKList, sumCDList))
            sumOrderedKLDict = OrderedDict(sorted(sumKLDict.items()))
            sumPercSmall[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, sumOrderedKLDict)
            sumPercLarge[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, sumOrderedKLDict)

            minKLDict = dict(zip(minKList, minCDList))
            minOrderedKLDict = OrderedDict(sorted(minKLDict.items()))
            minPercSmall[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, minOrderedKLDict)
            minPercLarge[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, minOrderedKLDict)

            maxKLDict = dict(zip(maxKList, maxCDList))
            maxOrderedKLDict = OrderedDict(sorted(maxKLDict.items()))
            maxPercSmall[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, maxOrderedKLDict)
            maxPercLarge[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, maxOrderedKLDict)
        
        # SUM UP REPEATS FOR ALL THE MAIN STATISTICS
        print(f"\naLambda: {aLambda[trial, INDEX_COUNT]}")
        print(f"sumLambda: {sumLambda[trial, INDEX_COUNT]}")
        aLambda[trial, INDEX_COUNT] = fmean(sumLambda[trial, INDEX_COUNT])
        aSum[trial, INDEX_COUNT] = fmean(minSum[trial, INDEX_COUNT])
        aPairLambda[trial, INDEX_COUNT] = fmean(minPairLambda[trial, INDEX_COUNT])
        aPairEst[trial, INDEX_COUNT] = fmean(minPairEst[trial, INDEX_COUNT])
        bPairLambda[trial, INDEX_COUNT] = fmean(maxPairLambda[trial, INDEX_COUNT])
        bPairEst[trial, INDEX_COUNT] = fmean(maxPairEst[trial, INDEX_COUNT])

        aPercSmall[trial, INDEX_COUNT] = fmean(tPercSmall[trial, INDEX_COUNT])
        aPercLarge[trial, INDEX_COUNT] = fmean(tPercLarge[trial, INDEX_COUNT])
        bPercSmall[trial, INDEX_COUNT] = fmean(sumPercSmall[trial, INDEX_COUNT])
        bPercLarge[trial, INDEX_COUNT] = fmean(sumPercLarge[trial, INDEX_COUNT])
        cPercSmall[trial, INDEX_COUNT] = fmean(minPercSmall[trial, INDEX_COUNT])
        cPercLarge[trial, INDEX_COUNT] = fmean(minPercLarge[trial, INDEX_COUNT])
        dPercSmall[trial, INDEX_COUNT] = fmean(maxPercSmall[trial, INDEX_COUNT])
        dPercLarge[trial, INDEX_COUNT] = fmean(maxPercLarge[trial, INDEX_COUNT])

        statsfile = open(f"emnist_{trialset[trial]}_noise_t_{T}.txt", "w", encoding = 'utf-8')
        statsfile.write(f"EMNIST: Laplace Noise in Middle, no Monte Carlo, T = {T}\n")
        statsfile.write(f"Optimal Lambda {round(aLambda[trial, INDEX_COUNT], 4)} for Sum {round(aSum[trial, INDEX_COUNT], 4)}\n\n")

        statsfile.write(f"Digit Pair with Min Exact KLD: {minPair}\n")
        statsfile.write(f"Optimal Lambda {round(aPairLambda[trial, INDEX_COUNT], 4)} for Estimate {round(aPairEst[trial, INDEX_COUNT], 4)}\n\n")

        statsfile.write(f"Digit Pair with Max Exact KLD: {maxPair}\n")
        statsfile.write(f"Optimal Lambda {round(bPairLambda[trial, INDEX_COUNT], 4)} for Estimate {round(bPairEst[trial, INDEX_COUNT], 4)}\n\n")

        statsfile.write(f"Smallest 10% exact KLD -> smaller half true dist ranking: {round(aPercSmall[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Largest 10% exact KLD -> larger half true dist ranking: {round(aPercLarge[trial, INDEX_COUNT], 1)}%\n\n")
        
        statsfile.write(f"Smallest 10% exact KLD -> smaller half sum ranking: {round(bPercSmall[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Largest 10% exact KLD -> larger half sum ranking: {round(bPercLarge[trial, INDEX_COUNT], 1)}%\n\n")

        statsfile.write(f"Smallest 10% exact KLD -> smaller half min pair ranking: {round(cPercSmall[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Largest 10% exact KLD -> larger half min pair ranking: {round(cPercLarge[trial, INDEX_COUNT], 1)}%\n\n")

        statsfile.write(f"Smallest 10% exact KLD -> smaller half max pair ranking: {round(dPercSmall[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Largest 10% exact KLD -> larger half max pair ranking: {round(dPercLarge[trial, INDEX_COUNT], 1)}%\n\n")

    INDEX_COUNT = INDEX_COUNT + 1

# PLOT LAMBDAS FOR EACH T
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
plt.savefig("Emnist_t_mid_lambda_sum.png")
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
plt.savefig("Emnist_t_mid_lambda_min_max.png")
plt.clf()

# PLOT SUM / ESTIMATES FOR EACH T
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
plt.savefig("Emnist_t_mid_est_sum.png")
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
plt.savefig("Emnist_t_mid_est_min_max.png")
plt.clf()

# PLOT RANKING PRESERVATIONS FOR EACH T
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
# plt.errorbar(Tset, aPercLarge[7], yerr = np.std(aPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest10%')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for true distribution")
plt.savefig("Emnist_t_mid_perc_ratio.png")
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
plt.savefig("Emnist_t_mid_perc_sum.png")
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
# plt.errorbar(Tset, cPercSmall[5], yerr = np.std(cPercSmall[5], axis = 0),  = 'tab:green', marker = 'o', label = 'end lap mc: smallest 10%')
# plt.errorbar(Tset, cPercLarge[5], yerr = np.std(cPercLarge[5], axis = 0), color = 'tab:green', marker = 'x', label = 'end lap mc: largest 10%')
# plt.errorbar(Tset, cPercSmall[6], yerr = np.std(cPercSmall[6], axis = 0), color = 'tab:red', marker = 'o', label = 'end gauss: smallest 10%')
# plt.errorbar(Tset, cPercLarge[6], yerr = np.std(cPercLarge[6], axis = 0), color = 'tab:red', marker = 'x', label = 'end gauss: largest 10%')
# plt.errorbar(Tset, cPercSmall[7], yerr = np.std(cPercSmall[7], axis = 0), color = 'tab:pink', marker = 'o', label = 'end gauss mc: smallest 10%')
# plt.errorbar(Tset, cPercLarge[7], yerr = np.std(cPercLarge[7], axis = 0), color = 'tab:pink', marker = 'x', label = 'end gauss mc: largest 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel(f"% staying in smaller/larger half")
plt.title("Ranking preservation for error of unbiased estimator (min pair)")
plt.savefig("Emnist_t_mid_perc_min.png")
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
plt.savefig("Emnist_t_mid_perc_max.png")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
