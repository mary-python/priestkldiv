"""Modules provide various time-related functions, generate pseudo-random numbers,
compute the natural logarithm of a number, remember the order in which items are added,
have cool visual feedback of the current throughput, create static, animated,
and interactive visualisations, compute the mean of a list quickly and accurately,
provide functionality to automatically download and cache the EMNIST dataset,
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

# LISTS OF THE EPS VALUES AND TRIALS THAT WILL BE RUN
epsset = [0.001, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]
trialset = ["mid_lap", "mid_lap_mc", "mid_gauss", "mid_gauss_mc", "end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc"]
ES = len(epsset)
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
tPercTop = np.zeros((TS, ES, R))
tPercBottom = np.zeros((TS, ES, R))
sumPercTop = np.zeros((TS, ES, R))
sumPercBottom = np.zeros((TS, ES, R))
minPercTop = np.zeros((TS, ES, R))
minPercBottom = np.zeros((TS, ES, R))
maxPercTop = np.zeros((TS, ES, R))
maxPercBottom = np.zeros((TS, ES, R))

aPercTop = np.zeros((TS, ES))
aPercBottom = np.zeros((TS, ES))
bPercTop = np.zeros((TS, ES))
bPercBottom = np.zeros((TS, ES))
cPercTop = np.zeros((TS, ES))
cPercBottom = np.zeros((TS, ES))
dPercTop = np.zeros((TS, ES))
dPercBottom = np.zeros((TS, ES))

# for trial in range(8):
for trial in range(4):

    print(f"\nTrial {trial + 1}: {trialset[trial]}")
    INDEX_COUNT = 0

    for eps in epsset:
        for rep in range(R):

            print(f"Trial {trial + 1}: epsilon = {eps}, repeat = {rep + 1}...")

            # STORES FOR EXACT KLD
            KLDiv = np.zeros((10, 10, U))
            KList = []
            CDList = []

            # STORES FOR ESTIMATED KLD
            eKLDiv = np.zeros((10, 10, E, R))
            eKList = []

            # STORES FOR RATIO BETWEEN KLDS AND TRUE DISTRIBUTION
            rKList = []
            rCDList = []
            tKList = []

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
                        rCDList.append((C, D))

                        # COMPUTE TRUE DISTRIBUTION
                        trueDist = abs(sum(eKLDiv[C, D, j]) * log(ratio))
                        tKList.append(trueDist)

                        # WAIT UNTIL FINAL DIGIT PAIR (9, 8) TO ANALYSE EXACT KLD LIST
                        if C == 9 and D == 8:

                            low = 0
                            high = 1
                            mid = 0.5

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

                            # EXTRACT MIN PAIR BY ABSOLUTE VALUE OF EXACT KL DIVERGENCE
                            absKList = [abs(kl) for kl in KList]
                            minKList = sorted(absKList)
                            minAbs = minKList[0]
                            minIndex = KList.index(minAbs)
                            minPair = CDList[minIndex]
                            MIN_COUNT = 1

                            # IF MIN PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT SMALLEST
                            while minPair not in rCDList:        
                                minAbs = minKList[MIN_COUNT]
                                minIndex = KList.index(minAbs)
                                minPair = CDList[minIndex]
                                MIN_COUNT = MIN_COUNT + 1

                            midMinIndex = halfCDList.index(minPair)
                            minPairEst[trial, INDEX_COUNT, rep] = halfKList[midMinIndex]

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

                                lowMinKL = min_max(low, rKList, tKList, lowKList, lowCDList, minPair)
                                highMinKL = min_max(high, rKList, tKList, highKList, highCDList, minPair)

                                if abs(lowMinKL) < abs(highMinKL):
                                    high = mid
                                else:
                                    low = mid

                                mid = (0.5*abs((high - low))) + low
                                minPairEst[trial, INDEX_COUNT, rep] = min_max(mid, rKList, tKList, minKList, minCDList, minPair)

                            minPairLambda[trial, INDEX_COUNT, rep] = mid

                            # EXTRACT MAX PAIR BY REVERSING EXACT KL DIVERGENCE LIST
                            maxKList = sorted(absKList, reverse = True)
                            maxAbs = maxKList[0]
                            maxIndex = KList.index(maxAbs)
                            maxPair = CDList[maxIndex]
                            MAX_COUNT = 1

                            # IF MAX PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT LARGEST
                            while maxPair not in rCDList:        
                                maxAbs = maxKList[MAX_COUNT]
                                maxIndex = KList.index(maxAbs)
                                maxPair = CDList[maxIndex]
                                MAX_COUNT = MAX_COUNT + 1

                            midMaxIndex = halfCDList.index(maxPair)
                            maxPairEst[trial, INDEX_COUNT, rep] = halfKList[midMaxIndex]

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
                """Do top/bottom 10% in exact KLD remain in top/bottom half of estimator?"""
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

            # EXACT KL DIVERGENCE IS IDENTICAL FOR ALL TRIALS, EPS AND REPEATS
            if trial == 0 and eps == 0.001 and rep == 0:
                KLDict = dict(zip(KList, CDList))
                orderedKLDict = OrderedDict(sorted(KLDict.items()))
                orderfile = open("emnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
                orderfile.write("EMNIST: Exact KL Divergence In Order\n")
                orderfile.write("Smaller corresponds to more similar digits\n\n")

            for i in orderedKLDict:
                orderfile.write(f"{i} : {orderedKLDict[i]}\n")

            # COMPUTE RANKING PRESERVATION STATISTICS FOR EACH REPEAT
            tKLDict = dict(zip(tKList, rCDList))
            tOrderedKLDict = OrderedDict(sorted(tKLDict.items()))
            tPercTop[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, tOrderedKLDict)
            tPercBottom[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, tOrderedKLDict)

            sumKLDict = dict(zip(sumKList, rCDList))
            sumOrderedKLDict = OrderedDict(sorted(sumKLDict.items()))
            sumPercTop[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, sumOrderedKLDict)
            sumPercBottom[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, sumOrderedKLDict)

            minKLDict = dict(zip(minKList, rCDList))
            minOrderedKLDict = OrderedDict(sorted(minKLDict.items()))
            minPercTop[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, minOrderedKLDict)
            minPercBottom[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, minOrderedKLDict)

            maxKLDict = dict(zip(maxKList, CDList))
            maxOrderedKLDict = OrderedDict(sorted(maxKLDict.items()))
            maxPercTop[trial, INDEX_COUNT, rep] = rank_pres(0, orderedKLDict, maxOrderedKLDict)
            maxPercBottom[trial, INDEX_COUNT, rep] = rank_pres(1, orderedKLDict, maxOrderedKLDict)
        
        # SUM UP REPEATS FOR ALL THE MAIN STATISTICS
        aLambda[trial, INDEX_COUNT] = fmean(sumLambda[trial, INDEX_COUNT])
        aSum[trial, INDEX_COUNT] = fmean(minSum[trial, INDEX_COUNT])
        aPairLambda[trial, INDEX_COUNT] = fmean(minPairLambda[trial, INDEX_COUNT])
        aPairEst[trial, INDEX_COUNT] = fmean(minPairEst[trial, INDEX_COUNT])
        bPairLambda[trial, INDEX_COUNT] = fmean(maxPairLambda[trial, INDEX_COUNT])
        bPairEst[trial, INDEX_COUNT] = fmean(maxPairEst[trial, INDEX_COUNT])

        aPercTop[trial, INDEX_COUNT] = fmean(tPercTop[trial, INDEX_COUNT])
        aPercBottom[trial, INDEX_COUNT] = fmean(tPercBottom[trial, INDEX_COUNT])
        bPercTop[trial, INDEX_COUNT] = fmean(sumPercTop[trial, INDEX_COUNT])
        bPercBottom[trial, INDEX_COUNT] = fmean(sumPercBottom[trial, INDEX_COUNT])
        cPercTop[trial, INDEX_COUNT] = fmean(minPercTop[trial, INDEX_COUNT])
        cPercBottom[trial, INDEX_COUNT] = fmean(minPercBottom[trial, INDEX_COUNT])
        dPercTop[trial, INDEX_COUNT] = fmean(maxPercTop[trial, INDEX_COUNT])
        dPercBottom[trial, INDEX_COUNT] = fmean(maxPercBottom[trial, INDEX_COUNT])

        statsfile = open(f"emnist_{trialset[trial]}_noise_eps_{eps}.txt", "w", encoding = 'utf-8')
        statsfile.write(f"EMNIST: Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
        statsfile.write(f"Optimal Lambda {round(aLambda[trial, INDEX_COUNT], 4)} for Sum {round(aSum[trial, INDEX_COUNT], 4)}\n\n")

        statsfile.write(f"Digit Pair with Min Exact KLD: {minPair}\n")
        statsfile.write(f"Optimal Lambda {round(aPairLambda[trial, INDEX_COUNT], 4)} for Estimate {round(aPairEst[trial, INDEX_COUNT], 4)}\n\n")

        statsfile.write(f"Digit Pair with Max Exact KLD: {maxPair}\n")
        statsfile.write(f"Optimal Lambda {round(bPairLambda[trial, INDEX_COUNT], 4)} for Estimate {round(bPairEst[trial, INDEX_COUNT], 4)}\n\n")

        statsfile.write(f"Top 10% exact KLD -> top half true dist ranking: {round(aPercTop[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Bottom 10% exact KLD -> bottom true dist ranking: {round(aPercBottom[trial, INDEX_COUNT], 1)}%\n\n")
        
        statsfile.write(f"Top 10% exact KLD -> top half sum ranking: {round(bPercTop[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Bottom 10% exact KLD -> bottom half sum ranking: {round(bPercBottom[trial, INDEX_COUNT], 1)}%\n\n")

        statsfile.write(f"Top 10% exact KLD -> top half min pair ranking: {round(cPercTop[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Bottom 10% exact KLD -> bottom half min pair ranking: {round(cPercBottom[trial, INDEX_COUNT], 1)}%\n\n")

        statsfile.write(f"Top 10% exact KLD -> top half max pair ranking: {round(dPercTop[trial, INDEX_COUNT], 1)}%\n")
        statsfile.write(f"Bottom 10% exact KLD -> bottom half max pair ranking: {round(dPercBottom[trial, INDEX_COUNT], 1)}%\n\n")

        INDEX_COUNT = INDEX_COUNT + 1

# PLOT LAMBDAS FOR EACH EPSILON
plt.plot(epsset, aLambda[0], color = 'tab:brown', label = 'mid lap')
plt.plot(epsset, aLambda[1], color = 'tab:purple', label = 'mid lap mc')
plt.plot(epsset, aLambda[2], color = 'tab:blue', label = 'mid gauss')
plt.plot(epsset, aLambda[3], color = 'tab:cyan', label = 'mid gauss mc')
# plt.plot(epsset, aLambda[4], color = 'tab:olive', label = 'end lap')
# plt.plot(epsset, aLambda[5], color = 'tab:green', label = 'end lap mc')
# plt.plot(epsset, aLambda[6], color = 'tab:red', label = 'end gauss')
# plt.plot(epsset, aLambda[7], color = 'tab:pink', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How epsilon affects lambda (sum)")
plt.savefig("Emnist_mid_lambda_sum.png")
plt.clf()

plt.plot(epsset, aPairLambda[0], color = 'tab:brown', label = 'mid lap: min')
plt.plot(epsset, bPairLambda[0], color = 'tab:brown', label = 'mid lap: max')
plt.plot(epsset, aPairLambda[1], color = 'tab:purple', label = 'mid lap mc: min')
plt.plot(epsset, bPairLambda[1], color = 'tab:purple', label = 'mid lap mc: max')
plt.plot(epsset, aPairLambda[2], color = 'tab:blue', label = 'mid gauss: min')
plt.plot(epsset, bPairLambda[2], color = 'tab:blue', label = 'mid gauss: max')
plt.plot(epsset, aPairLambda[3], color = 'tab:cyan', label = 'mid gauss mc: min')
plt.plot(epsset, bPairLambda[3], color = 'tab:cyan', label = 'mid gauss mc: max')
# plt.plot(epsset, aPairLambda[4], color = 'tab:olive', label = 'end lap: min')
# plt.plot(epsset, bPairLambda[4], color = 'tab:olive', label = 'end lap: max')
# plt.plot(epsset, aPairLambda[5], color = 'tab:green', label = 'end lap mc: min')
# plt.plot(epsset, bPairLambda[5], color = 'tab:green', label = 'end lap mc: max')
# plt.plot(epsset, aPairLambda[6], color = 'tab:red', label = 'end gauss: min')
# plt.plot(epsset, bPairLambda[6], color = 'tab:red', label = 'end gauss: max')
# plt.plot(epsset, aPairLambda[7], color = 'tab:pink', label = 'end gauss mc: min')
# plt.plot(epsset, bPairLambda[7], color = 'tab:pink', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How epsilon affects lambda (min/max pair)")
plt.savefig("Emnist_mid_lambda_min_max.png")
plt.clf()

# PLOT SUM / ESTIMATES FOR EACH EPSILON
plt.plot(epsset, aSum[0], color = 'tab:brown', label = 'mid lap')
plt.plot(epsset, aSum[1], color = 'tab:purple', label = 'mid lap mc')
plt.plot(epsset, aSum[2], color = 'tab:blue', label = 'mid gauss')
plt.plot(epsset, aSum[3], color = 'tab:cyan', label = 'mid gauss mc')
# plt.plot(epsset, aSum[4], color = 'tab:olive', label = 'end lap')
# plt.plot(epsset, aSum[5], color = 'tab:green', label = 'end lap mc')
# plt.plot(epsset, aSum[6], color = 'tab:red', label = 'end gauss')
# plt.plot(epsset, aSum[7], color = 'tab:pink', label = 'end gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (sum)")
plt.title("How epsilon affects error of unbiased estimator (sum)")
plt.savefig("Emnist_mid_est_sum.png")
plt.clf()

plt.plot(epsset, aPairEst[0], color = 'tab:brown', label = 'mid lap: min')
plt.plot(epsset, bPairEst[0], color = 'tab:brown', label = 'mid lap: max')
plt.plot(epsset, aPairEst[1], color = 'tab:purple', label = 'mid lap mc: min')
plt.plot(epsset, bPairEst[1], color = 'tab:purple', label = 'mid lap mc: max')
plt.plot(epsset, aPairEst[2], color = 'tab:blue', label = 'mid gauss: min')
plt.plot(epsset, bPairEst[2], color = 'tab:blue', label = 'mid gauss: max')
plt.plot(epsset, aPairEst[3], color = 'tab:cyan', label = 'mid gauss mc: min')
plt.plot(epsset, bPairEst[3], color = 'tab:cyan', label = 'mid gauss mc: max')
# plt.plot(epsset, aPairEst[4], color = 'tab:olive', label = 'end lap: min')
# plt.plot(epsset, bPairEst[4], color = 'tab:olive', label = 'end lap: max')
# plt.plot(epsset, aPairEst[5], color = 'tab:green', label = 'end lap mc: min')
# plt.plot(epsset, bPairEst[5], color = 'tab:green', label = 'end lap mc: max')
# plt.plot(epsset, aPairEst[6], color = 'tab:red', label = 'end gauss: min')
# plt.plot(epsset, bPairEst[6], color = 'tab:red', label = 'end gauss: max')
# plt.plot(epsset, aPairEst[7], color = 'tab:pink', label = 'end gauss mc: min')
# plt.plot(epsset, bPairEst[7], color = 'tab:pink', label = 'end gauss mc: max')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel("Error of unbiased estimator (min/max pair)")
plt.title("How epsilon affects error of unbiased estimator (min/max pair)")
plt.savefig("Emnist_mid_est_min_max.png")
plt.clf()

# PLOT RANKING PRESERVATIONS FOR EACH EPSILON
plt.plot(epsset, aPercTop[0], color = 'tab:brown', label = 'mid lap: top 10%')
plt.plot(epsset, aPercBottom[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
plt.plot(epsset, aPercTop[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
plt.plot(epsset, aPercBottom[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
plt.plot(epsset, aPercTop[2], color = 'tab:blue', label = 'mid gauss: top 10%')
plt.plot(epsset, aPercBottom[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
plt.plot(epsset, aPercTop[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
plt.plot(epsset, aPercBottom[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
# plt.plot(epsset, aPercTop[4], color = 'tab:olive', label = 'end lap: top 10%')
# plt.plot(epsset, aPercBottom[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, aPercTop[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, aPercBottom[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, aPercTop[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, aPercBottom[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, aPercTop[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, aPercBottom[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top/bottom half")
plt.title("Ranking preservation for true distribution")
plt.savefig("Emnist_mid_perc_ratio.png")
plt.clf()

plt.plot(epsset, bPercTop[0], color = 'tab:brown', label = 'mid lap: top 10%')
plt.plot(epsset, bPercBottom[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
plt.plot(epsset, bPercTop[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
plt.plot(epsset, bPercBottom[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
plt.plot(epsset, bPercTop[2], color = 'tab:blue', label = 'mid gauss: top 10%')
plt.plot(epsset, bPercBottom[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
plt.plot(epsset, bPercTop[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
plt.plot(epsset, bPercBottom[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
# plt.plot(epsset, bPercTop[4], color = 'tab:olive', label = 'end lap: top 10%')
# plt.plot(epsset, bPercBottom[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, bPercTop[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, bPercBottom[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, bPercTop[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, bPercBottom[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, bPercTop[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, bPercBottom[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top / bottom half")
plt.title("Ranking preservation for error of unbiased estimator (sum)")
plt.savefig("Emnist_mid_perc_sum.png")
plt.clf()

plt.plot(epsset, cPercTop[0], color = 'tab:brown', label = 'mid lap: top 10%')
plt.plot(epsset, cPercBottom[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
plt.plot(epsset, cPercTop[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
plt.plot(epsset, cPercBottom[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
plt.plot(epsset, cPercTop[2], color = 'tab:blue', label = 'mid gauss: top 10%')
plt.plot(epsset, cPercBottom[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
plt.plot(epsset, cPercTop[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
plt.plot(epsset, cPercBottom[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
# plt.plot(epsset, cPercTop[4], color = 'tab:olive', label = 'end lap: top 10%')
# plt.plot(epsset, cPercBottom[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, cPercTop[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, cPercBottom[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, cPercTop[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, cPercBottom[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, cPercTop[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, cPercBottom[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top / bottom half")
plt.title("Ranking preservation for error of unbiased estimator (min pair)")
plt.savefig("Emnist_mid_perc_min.png")
plt.clf()

plt.plot(epsset, dPercTop[0], color = 'tab:brown', label = 'mid lap: top 10%')
plt.plot(epsset, dPercBottom[0], color = 'tab:brown', label = 'mid lap: bottom 10%')
plt.plot(epsset, dPercTop[1], color = 'tab:purple', label = 'mid lap mc: top 10%')
plt.plot(epsset, dPercBottom[1], color = 'tab:purple', label = 'mid lap mc: bottom 10%')
plt.plot(epsset, dPercTop[2], color = 'tab:blue', label = 'mid gauss: top 10%')
plt.plot(epsset, dPercBottom[2], color = 'tab:blue', label = 'mid gauss: bottom 10%')
plt.plot(epsset, dPercTop[3], color = 'tab:cyan', label = 'mid gauss mc: top 10%')
plt.plot(epsset, dPercBottom[3], color = 'tab:cyan', label = 'mid gauss mc: bottom 10%')
# plt.plot(epsset, dPercTop[4], color = 'tab:olive', label = 'end lap: top 10%')
# plt.plot(epsset, dPercBottom[4], color = 'tab:olive', label = 'end lap: bottom 10%')
# plt.plot(epsset, dPercTop[5], color = 'tab:green', label = 'end lap mc: top 10%')
# plt.plot(epsset, dPercBottom[5], color = 'tab:green', label = 'end lap mc: bottom 10%')
# plt.plot(epsset, dPercTop[6], color = 'tab:red', label = 'end gauss: top 10%')
# plt.plot(epsset, dPercBottom[6], color = 'tab:red', label = 'end gauss: bottom 10%')
# plt.plot(epsset, dPercTop[7], color = 'tab:pink', label = 'end gauss mc: top 10%')
# plt.plot(epsset, dPercBottom[7], color = 'tab:pink', label = 'end gauss mc: bottom 10%')
plt.legend(loc = 'best')
plt.xlabel("Value of epsilon")
plt.ylabel(f"% staying in top / bottom half")
plt.title("Ranking preservation for error of unbiased estimator (max pair)")
plt.savefig("Emnist_mid_perc_max.png")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
