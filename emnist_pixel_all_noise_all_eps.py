"""Modules provide various time-related functions, generate pseudo-random numbers,
compute the natural logarithm of a number, remember the order in which items are added,
have cool visual feedback of the current throughput, create static, animated, and
interactive visualisations, provide functionality to automatically download and cache the
EMNIST dataset, work with arrays, and carry out fast numerical computations in Python."""
import time
import random
from math import log
from collections import OrderedDict
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
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

# NUMBER OF REPEATS COVERS APPROX 5% OF IMAGES
T = 1400

# STORE T IMAGES CORRESPONDING TO EACH DIGIT
sampleImSet = np.zeros((10, T, 4, 4))
sampleImList = np.zeros((14000, 4, 4))
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
sizeUniqueImList = len(uniqueImList)

# DOMAIN FOR EACH DIGIT DISTRIBUTION IS NUMBER OF UNIQUE IMAGES
U = 207

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
DTA = 0.1
A = 0
R = 10

# STORES FOR SUM, MIN/MAX KLD AND LAMBDA
minSum = np.zeros((8, 12))
minPairEst = np.zeros((8, 12))
maxPairEst = np.zeros((8, 12))
sumLambda = np.zeros((8, 12))
minPairLambda = np.zeros((8, 12))
maxPairLambda = np.zeros((8, 12))

# STORES FOR RANKING PRESERVATION ANALYSIS
rPercTopKLD = np.zeros((8, 12))
rPercBottomKLD = np.zeros((8, 12))
sumPercTopKLD = np.zeros((8, 12))
sumPercBottomKLD = np.zeros((8, 12))
minPercTopKLD = np.zeros((8, 12))
minPercBottomKLD = np.zeros((8, 12))
maxPercTopKLD = np.zeros((8, 12))
maxPercBottomKLD = np.zeros((8, 12))

epsset = [0.001, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]
trialset = ["mid_lap", "mid_lap_mc", "mid_gauss", "mid_gauss_mc", "end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc"]

# for trial in range(8):
for trial in range(4, 5):

    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    INDEX_COUNT = 0

    for eps in epsset:

        print(f"Trial {trial + 1}: epsilon = {eps}...")

        # STORES FOR EXACT KLD
        KLDiv = np.zeros((10, 10, U))
        sumKLDiv = np.zeros((10, 10))
        KList = []
        CDList = []

        # STORES FOR ESTIMATED KLD
        eKLDiv = np.zeros((10, 10, E))
        eSumKLDiv = np.zeros((10, 10))
        eKList = []
        eCDList = []

        # STORES FOR RATIO BETWEEN EXACT AND ESTIMATED KLD
        rKList = []
        rCDList = []

        # STORES FOR UNBIASED ESTIMATE OF KLD
        zeroKList = []
        zeroCDList = []
        oneKList = []
        oneCDList = []
        halfKList = []
        halfCDList = []

        # OPTION 1A: QUERYING ENTIRE DISTRIBUTION
        if trial % 2 == 0:
            b1 = log(2) / eps

        # OPTION 1B: MONTE CARLO SAMPLING
        else:
            b1 = (1 + log(2)) / eps

        b2 = (2*((log(1.25))/DTA)*b1) / eps

        # OPTION 2A: ADD LAPLACE NOISE
        if trial % 4 == 0 or trial % 4 == 1:
            noiseLG = tfp.distributions.Laplace(loc = A, scale = b1)
        
        # OPTION 2B: ADD GAUSSIAN NOISE
        else:
            noiseLG = tfp.distributions.Normal(loc = A, scale = b2)

        def unbias_est(lda, rklist, lklist, lcdlist, tnl):
            """Compute sum of unbiased estimators corresponding to all pairs."""
            count = 1

            for rat in rklist:
                lest = ((lda * (rat - 1)) - log(rat)) / T

                # OPTION 3B: ADD NOISE AT END
                if trial >= 4:
                    for k in range(0, R):
                        tnl = tnl + (noiseLG.sample(sample_shape = (1,)).numpy()[0])
       
                    # COMPUTE AVERAGE OF R POSSIBLE NOISE TERMS
                    lest = lest + (tnl / R)

                if lest != 0.0:
                    lklist.append(lest)

                    c = count // 10
                    d = count % 10

                    lcdlist.append((c, d))

                    if c == d + 1:
                        count = count + 2
                    else:
                        count = count + 1

            return sum(lklist)
    
        def min_max(lda, rklist, lklist, lcdlist, mp, tnl):
            """Compute unbiased estimator corresponding to min or max pair."""
            count = 1

            for rat in rklist:
                lest = ((lda * (rat - 1)) - log(rat)) / T

                # OPTION 3B: ADD NOISE AT END
                if trial >= 4:
                    for k in range(0, R):
                        tnl = tnl + (noiseLG.sample(sample_shape = (1,)).numpy()[0])
       
                    # COMPUTE AVERAGE OF R POSSIBLE NOISE TERMS
                    lest = lest + (tnl / R)

                if lest != 0.0:
                    lklist.append(lest)

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
                        totalNoiseLG = 0

                        for k in range(0, R):
                            totalNoiseLG = totalNoiseLG + (noiseLG.sample(sample_shape = (1,)))
            
                        # COMPUTE AVERAGE OF R POSSIBLE NOISE TERMS
                        avNoiseLG = totalNoiseLG / R
                        eKLDiv[C, D, j] = eKLDiv[C, D, j] + avNoiseLG

                # ELIMINATE ALL ZERO VALUES WHEN DIGITS ARE IDENTICAL
                if sum(KLDiv[C, D]) != 0.0:
                    KList.append(sum(KLDiv[C, D]))
                    CDList.append((C, D))

                # STILL NEED BELOW CONDITION TO AVOID ZERO ERROR IN RATIO
                if sum(eKLDiv[C, D]) != 0.0:
                    eKList.append(sum(eKLDiv[C, D]))
                    eCDList.append((C, D))

                # COMPUTE RATIO BETWEEN EXACT AND ESTIMATED KLD
                ratio = abs(sum(eKLDiv[C, D]) / sum(KLDiv[C, D]))

                # ELIMINATE ALL DIVIDE BY ZERO ERRORS
                if ratio != 0.0 and sum(KLDiv[C, D]) != 0.0:
                    rKList.append(ratio)
                    rCDList.append((C, D))

                    # WAIT UNTIL FINAL DIGIT PAIR (8, 9) TO ANALYSE EXACT KLD LIST
                    if C == 9 and D == 8:

                        low = 0
                        high = 1
                        mid = 0.5

                        # COMPUTE UNBIASED ESTIMATORS WITH LAMBDA 0, 1, 0.5 THEN BINARY SEARCH
                        lowSum = unbias_est(low, rKList, zeroKList, zeroCDList, 0)
                        highSum = unbias_est(high, rKList, oneKList, oneCDList, 0)
                        minSum[trial, INDEX_COUNT] = unbias_est(mid, rKList, halfKList, halfCDList, 0)

                        # TOLERANCE BETWEEN BINARY SEARCH LIMITS ALWAYS GETS SMALL ENOUGH
                        while abs(high - low) > 0.00000001:

                            lowKList = []
                            lowCDList = []
                            highKList = []
                            highCDList = []
                            sumKList = []
                            sumCDList = []

                            lowSum = unbias_est(low, rKList, lowKList, lowCDList, 0)
                            highSum = unbias_est(high, rKList, highKList, highCDList, 0)

                            # REDUCE / INCREASE BINARY SEARCH LIMIT DEPENDING ON ABSOLUTE VALUE
                            if abs(lowSum) < abs(highSum):
                                high = mid
                            else:
                                low = mid

                            # SET NEW MIDPOINT
                            mid = (0.5*abs((high - low))) + low
                            minSum[trial, INDEX_COUNT] = unbias_est(mid, rKList, sumKList, sumCDList, 0)

                        sumLambda[trial, INDEX_COUNT] = mid

                        # EXTRACT MIN PAIR BY ABSOLUTE VALUE OF EXACT KL DIVERGENCE
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
                        minPairEst[trial, INDEX_COUNT] = halfKList[midMinIndex]

                        low = 0
                        high = 1
                        mid = 0.5

                        # FIND OPTIMAL LAMBDA FOR MIN PAIR
                        while abs(high - low) > 0.00000001:

                            lowKList = []
                            lowCDList = []
                            highKList = []
                            highCDList = []
                            minKList = []
                            minCDList = []

                            lowMinKL = min_max(low, rKList, lowKList, lowCDList, minPair, 0)
                            highMinKL = min_max(high, rKList, highKList, highCDList, minPair, 0)

                            if abs(lowMinKL) < abs(highMinKL):
                                high = mid
                            else:
                                low = mid

                            mid = (0.5*abs((high - low))) + low
                            minPairEst[trial, INDEX_COUNT] = min_max(mid, rKList, minKList, minCDList, minPair, 0)

                        minPairLambda[trial, INDEX_COUNT] = mid

                        # EXTRACT MAX PAIR BY REVERSING EXACT KL DIVERGENCE LIST
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
                        maxPairEst[trial, INDEX_COUNT] = halfKList[midMaxIndex]

                        low = 0
                        high = 1
                        mid = 0.5

                        # FIND OPTIMAL LAMBDA FOR MAX PAIR
                        while abs(high - low) > 0.00000001:

                            lowKList = []
                            lowCDList = []
                            highKList = []
                            highCDList = []
                            maxKList = []
                            maxCDList = []

                            lowMaxKL = min_max(low, rKList, lowKList, lowCDList, maxPair, 0)
                            highMaxKL = min_max(high, rKList, highKList, highCDList, maxPair, 0)
                
                            if abs(lowMaxKL) < abs(highMaxKL):
                                high = mid
                            else:
                                low = mid

                            mid = (0.5*(abs(high - low))) + low
                            maxPairEst[trial, INDEX_COUNT] = min_max(mid, rKList, maxKList, maxCDList, maxPair, 0)

                        maxPairLambda[trial, INDEX_COUNT] = mid

        def rank_pres(bin, okld, rokld):
            """Do top/bottom 10% in exact KLD remain in top/bottom half of estimator?"""
            rows = 90
            num = 0

            if bin == 0: 
                dict = list(okld.values())[0 : int(rows / 10)]
                rdict = list(rokld.values())[0 : int(rows / 2)]
            else:
                dict = list(okld.values())[int(9*(rows / 10)) : rows]
                rdict = list(rokld.values())[int(rows / 2) : rows]

            for di in dict:
                for dj in rdict:    
                    if dj == di:
                        num = num + 1

            return 100*(num / int(rows/10))

        # CREATE ORDERED DICTIONARIES OF STORED KLD AND DIGITS
        KLDict = dict(zip(KList, CDList))
        orderedKLDict = OrderedDict(sorted(KLDict.items()))
        datafile = open("emnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
        datafile.write("EMNIST: Exact KL Divergence In Order\n")
        datafile.write("Smaller corresponds to more similar digits\n\n")

        for i in orderedKLDict:
            datafile.write(f"{i} : {orderedKLDict[i]}\n")

        eKLDict = dict(zip(eKList, eCDList))
        eOrderedKLDict = OrderedDict(sorted(eKLDict.items()))
        estfile = open(f"em_kld_{trialset[trial]}_noise_eps_{eps}_est.txt", "w", encoding = 'utf-8')
        estfile.write("EMNIST: Estimated KL Divergence In Order\n")
        estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
        estfile.write("Smaller corresponds to more similar digits\n\n")

        for j in eOrderedKLDict:
            estfile.write(f"{j} : {eOrderedKLDict[j]}\n")

        rKLDict = dict(zip(rKList, rCDList))
        rOrderedKLDict = OrderedDict(sorted(rKLDict.items()))
        ratiofile = open(f"em_kld_{trialset[trial]}_noise_eps_{eps}_ratio.txt", "w", encoding = 'utf-8')
        ratiofile.write("EMNIST: Ratio Between Exact KL Divergence And Estimator\n")
        ratiofile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
        ratiofile.write("Closer to 1 corresponds to a better estimate\n\n")

        rPercTopKLD[trial, INDEX_COUNT] = rank_pres(0, orderedKLDict, rOrderedKLDict)
        rPercBottomKLD[trial, INDEX_COUNT] = rank_pres(1, orderedKLDict, rOrderedKLDict)
        ratiofile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(rPercTopKLD[trial, INDEX_COUNT], 1)}%\n")
        ratiofile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(rPercBottomKLD[trial, INDEX_COUNT], 1)}%\n\n")

        for k in rOrderedKLDict:
            ratiofile.write(f"{k} : {rOrderedKLDict[k]}\n")

        sumKLDict = dict(zip(sumKList, sumCDList))
        sumOrderedKLDict = OrderedDict(sorted(sumKLDict.items()))
        sumfile = open(f"em_kld_{trialset[trial]}_noise_eps_{eps}_l3est.txt", "w", encoding = 'utf-8')
        sumfile.write("EMNIST: Unbiased Estimator Optimal Lambda for Sum\n")
        sumfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
        sumfile.write(f"Optimal Lambda {sumLambda[trial, INDEX_COUNT]} for Sum {minSum[trial, INDEX_COUNT]}\n\n")

        sumPercTopKLD[trial, INDEX_COUNT] = rank_pres(0, orderedKLDict, sumOrderedKLDict)
        sumPercBottomKLD[trial, INDEX_COUNT] = rank_pres(1, orderedKLDict, sumOrderedKLDict)
        sumfile.write(f"Top 10% exact KLD -> top half sum ranking: {round(sumPercTopKLD[trial, INDEX_COUNT], 1)}%\n")
        sumfile.write(f"Bottom 10% exact KLD -> bottom half sum ranking: {round(sumPercBottomKLD[trial, INDEX_COUNT], 1)}%\n\n")

        minKLDict = dict(zip(minKList, minCDList))
        minOrderedKLDict = OrderedDict(sorted(minKLDict.items()))
        minfile = open(f"em_kld_{trialset[trial]}_noise_eps_{eps}_l4est.txt", "w", encoding = 'utf-8')
        minfile.write("EMNIST: Unbiased Estimator Optimal Lambda Min Pair (7, 1)\n")
        minfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
        minfile.write(f"Optimal Lambda {minPairLambda[trial, INDEX_COUNT]} for Estimate {minPairEst[trial, INDEX_COUNT]}\n\n")

        minPercTopKLD[trial, INDEX_COUNT] = rank_pres(0, orderedKLDict, minOrderedKLDict)
        minPercBottomKLD[trial, INDEX_COUNT] = rank_pres(1, orderedKLDict, minOrderedKLDict)
        minfile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(minPercTopKLD[trial, INDEX_COUNT], 1)}%\n")
        minfile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(minPercBottomKLD[trial, INDEX_COUNT], 1)}%\n\n")

        maxKLDict = dict(zip(maxKList, maxCDList))
        maxOrderedKLDict = OrderedDict(sorted(maxKLDict.items()))
        maxfile = open(f"em_kld_{trialset[trial]}_noise_eps_{eps}_l5est.txt", "w", encoding = 'utf-8')
        maxfile.write("EMNIST: Unbiased Estimator Optimal Lambda Max Pair (6, 9)\n")
        maxfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
        maxfile.write(f"Optimal Lambda {maxPairLambda[trial, INDEX_COUNT]} for Estimate {maxPairEst[trial, INDEX_COUNT]}\n\n")

        maxPercTopKLD[trial, INDEX_COUNT] = rank_pres(0, orderedKLDict, maxOrderedKLDict)
        maxPercBottomKLD[trial, INDEX_COUNT] = rank_pres(1, orderedKLDict, maxOrderedKLDict)
        maxfile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(maxPercTopKLD[trial, INDEX_COUNT], 1)}%\n")
        maxfile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(maxPercBottomKLD[trial, INDEX_COUNT], 1)}%\n\n")

        INDEX_COUNT = INDEX_COUNT + 1

# PLOT LAMBDAS FOR EACH EPSILON
# plt.plot(epsset, sumLambda[0], color = 'tab:brown', label = 'mid lap')
# plt.plot(epsset, sumLambda[1], color = 'tab:purple', label = 'mid lap mc')
# plt.plot(epsset, sumLambda[2], color = 'tab:blue', label = 'mid gauss')
# plt.plot(epsset, sumLambda[3], color = 'tab:cyan', label = 'mid gauss mc')
plt.plot(epsset, sumLambda[4], color = 'tab:olive', label = 'end lap')
# plt.plot(epsset, sumLambda[5], color = 'tab:green', label = 'end lap mc')
# plt.plot(epsset, sumLambda[6], color = 'tab:red', label = 'end gauss')
# plt.plot(epsset, sumLambda[7], color = 'tab:pink', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise sum of unbiased estimators")
plt.title("How epsilon affects lambda (EMNIST)")
plt.savefig("Emnist_lambda_sum.png")
plt.clf()

# plt.plot(epsset, minPairLambda[0], color = 'tab:brown', label = 'mid lap: min')
# plt.plot(epsset, maxPairLambda[0], color = 'tab:brown', label = 'mid lap: max')
# plt.plot(epsset, minPairLambda[1], color = 'tab:purple', label = 'mid lap mc: min')
# plt.plot(epsset, maxPairLambda[1], color = 'tab:purple', label = 'mid lap mc: max')
# plt.plot(epsset, minPairLambda[2], color = 'tab:blue', label = 'mid gauss: min')
# plt.plot(epsset, maxPairLambda[2], color = 'tab:blue', label = 'mid gauss: max')
# plt.plot(epsset, minPairLambda[3], color = 'tab:cyan', label = 'mid gauss mc: min')
# plt.plot(epsset, maxPairLambda[3], color = 'tab:cyan', label = 'mid gauss mc: max')
plt.plot(epsset, minPairLambda[4], color = 'tab:olive', label = 'end lap: min')
plt.plot(epsset, maxPairLambda[4], color = 'tab:olive', label = 'end lap: max')
# plt.plot(epsset, minPairLambda[5], color = 'tab:green', label = 'end lap mc: min')
# plt.plot(epsset, maxPairLambda[5], color = 'tab:green', label = 'end lap mc: max')
# plt.plot(epsset, minPairLambda[6], color = 'tab:red', label = 'end gauss: min')
# plt.plot(epsset, maxPairLambda[6], color = 'tab:red', label = 'end gauss: max')
# plt.plot(epsset, minPairLambda[7], color = 'tab:pink', label = 'end gauss mc: min')
# plt.plot(epsset, maxPairLambda[7], color = 'tab:pink', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise unbiased estimator corresponding to min / max pair")
plt.title("How epsilon affects lambda (EMNIST)")
plt.savefig("Emnist_lambda_min_max.png")
plt.clf()

# PLOT SUM / ESTIMATES FOR EACH EPSILON
# plt.plot(epsset, minSum[0], color = 'tab:brown', label = 'mid lap')
# plt.plot(epsset, minSum[1], color = 'tab:purple', label = 'mid lap mc')
# plt.plot(epsset, minSum[2], color = 'tab:blue', label = 'mid gauss')
# plt.plot(epsset, minSum[3], color = 'tab:cyan', label = 'mid gauss mc')
plt.plot(epsset, minSum[4], color = 'tab:olive', label = 'end lap')
# plt.plot(epsset, minSum[5], color = 'tab:green', label = 'end lap mc')
# plt.plot(epsset, minSum[6], color = 'tab:red', label = 'end gauss')
# plt.plot(epsset, minSum[7], color = 'tab:pink', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Sum of unbiased estimators")
plt.title("How epsilon affects sum of unbiased estimators (EMNIST)")
plt.savefig("Emnist_est_sum.png")
plt.clf()

# plt.plot(epsset, minPairEst[0], color = 'tab:brown', label = 'mid lap: min')
# plt.plot(epsset, maxPairEst[0], color = 'tab:brown', label = 'mid lap: max')
# plt.plot(epsset, minPairEst[1], color = 'tab:purple', label = 'mid lap mc: min')
# plt.plot(epsset, maxPairEst[1], color = 'tab:purple', label = 'mid lap mc: max')
# plt.plot(epsset, minPairEst[2], color = 'tab:blue', label = 'mid gauss: min')
# plt.plot(epsset, maxPairEst[2], color = 'tab:blue', label = 'mid gauss: max')
# plt.plot(epsset, minPairEst[3], color = 'tab:cyan', label = 'mid gauss mc: min')
# plt.plot(epsset, maxPairEst[3], color = 'tab:cyan', label = 'mid gauss mc: max')
plt.plot(epsset, minPairEst[4], color = 'tab:olive', label = 'end lap: min')
plt.plot(epsset, maxPairEst[4], color = 'tab:olive', label = 'end lap: max')
# plt.plot(epsset, minPairEst[5], color = 'tab:green', label = 'end lap mc: min')
# plt.plot(epsset, maxPairEst[5], color = 'tab:green', label = 'end lap mc: max')
# plt.plot(epsset, minPairEst[6], color = 'tab:red', label = 'end gauss: min')
# plt.plot(epsset, maxPairEst[6], color = 'tab:red', label = 'end gauss: max')
# plt.plot(epsset, minPairEst[7], color = 'tab:pink', label = 'end gauss mc: min')
# plt.plot(epsset, maxPairEst[7], color = 'tab:pink', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Unbiased estimator corresponding to min / max pair")
plt.title("How epsilon affects unbiased estimator corresponding to min / max pair (EMNIST)")
plt.savefig("Emnist_est_min_max.png")
plt.clf()

# PLOT RANKING PRESERVATIONS FOR EACH EPSILON
# plt.plot(epsset, rPercTopKLD[0], color = 'tab:brown', label = 'mid lap: top 10%')
# plt.plot(epsset, rPercBottomKLD[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
# plt.plot(epsset, rPercTopKLD[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
# plt.plot(epsset, rPercBottomKLD[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
# plt.plot(epsset, rPercTopKLD[2], color = 'tab:blue', label = 'mid gauss: top 10%')
# plt.plot(epsset, rPercBottomKLD[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
# plt.plot(epsset, rPercTopKLD[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
# plt.plot(epsset, rPercBottomKLD[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
plt.plot(epsset, rPercTopKLD[4], color = 'tab:olive', label = 'end lap: top 10%')
plt.plot(epsset, rPercBottomKLD[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, rPercTopKLD[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, rPercBottomKLD[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, rPercTopKLD[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, rPercBottomKLD[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, rPercTopKLD[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, rPercBottomKLD[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top/bottom half")
plt.title("Ranking preservation for ratio between exact and estimated KLD")
plt.savefig("Emnist_perc_ratio.png")
plt.clf()

# plt.plot(epsset, sumPercTopKLD[0], color = 'tab:brown', label = 'mid lap: top 10%')
# plt.plot(epsset, sumPercBottomKLD[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
# plt.plot(epsset, sumPercTopKLD[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
# plt.plot(epsset, sumPercBottomKLD[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
# plt.plot(epsset, sumPercTopKLD[2], color = 'tab:blue', label = 'mid gauss: top 10%')
# plt.plot(epsset, sumPercBottomKLD[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
# plt.plot(epsset, sumPercTopKLD[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
# plt.plot(epsset, sumPercBottomKLD[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
plt.plot(epsset, sumPercTopKLD[4], color = 'tab:olive', label = 'end lap: top 10%')
plt.plot(epsset, sumPercBottomKLD[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, sumPercTopKLD[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, sumPercBottomKLD[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, sumPercTopKLD[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, sumPercBottomKLD[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, sumPercTopKLD[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, sumPercBottomKLD[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top / bottom half")
plt.title("Ranking preservation for sum of unbiased estimators (EMNIST)")
plt.savefig("Emnist_perc_sum.png")
plt.clf()

# plt.plot(epsset, minPercTopKLD[0], color = 'tab:brown', label = 'mid lap: top 10%')
# plt.plot(epsset, minPercBottomKLD[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
# plt.plot(epsset, minPercTopKLD[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
# plt.plot(epsset, minPercBottomKLD[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
# plt.plot(epsset, minPercTopKLD[2], color = 'tab:blue', label = 'mid gauss: top 10%')
# plt.plot(epsset, minPercBottomKLD[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
# plt.plot(epsset, minPercTopKLD[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
# plt.plot(epsset, minPercBottomKLD[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
plt.plot(epsset, minPercTopKLD[4], color = 'tab:olive', label = 'end lap: top 10%')
plt.plot(epsset, minPercBottomKLD[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, minPercTopKLD[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, minPercBottomKLD[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, minPercTopKLD[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, minPercBottomKLD[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, minPercTopKLD[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, minPercBottomKLD[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top / bottom half")
plt.title("Ranking preservation for unbiased estimator corresponding to min pair (EMNIST)")
plt.savefig("Emnist_perc_min.png")
plt.clf()

# plt.plot(epsset, maxPercTopKLD[0], color = 'tab:brown', label = 'mid lap: top 10%')
# plt.plot(epsset, maxPercBottomKLD[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
# plt.plot(epsset, maxPercTopKLD[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
# plt.plot(epsset, maxPercBottomKLD[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
# plt.plot(epsset, maxPercTopKLD[2], color = 'tab:blue', label = 'mid gauss: top 10%')
# plt.plot(epsset, maxPercBottomKLD[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
# plt.plot(epsset, maxPercTopKLD[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
# plt.plot(epsset, maxPercBottomKLD[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
plt.plot(epsset, maxPercTopKLD[4], color = 'tab:olive', label = 'end lap: top 10%')
plt.plot(epsset, maxPercBottomKLD[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, maxPercTopKLD[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, maxPercBottomKLD[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, maxPercTopKLD[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, maxPercBottomKLD[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, maxPercTopKLD[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, maxPercBottomKLD[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top / bottom half")
plt.title("Ranking preservation for unbiased estimator corresponding to max pair (EMNIST)")
plt.savefig("Emnist_perc_max.png")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
