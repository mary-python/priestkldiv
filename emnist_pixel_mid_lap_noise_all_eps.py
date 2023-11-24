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

epsset = [0.001, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]

for eps in epsset:

    print(f"\nComputing KL divergence for epsilon = {eps}...")

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
    lZeroKList = []
    lZeroCDList = []
    lOneKList = []
    lOneCDList = []
    lTwoKList = []
    lTwoCDList = []
    lThreeKList = []
    lThreeCDList = []

    # OPTION 1A: QUERYING ENTIRE DISTRIBUTION (B: MONTE CARLO SAMPLING)
    b1 = log(2) / eps
    b2 = (2*((log(1.25))/DTA)*b1) / eps

    # OPTION 2A: ADD LAPLACE NOISE (B: GAUSSIAN NOISE)
    noiseL = tfp.distributions.Laplace(loc = A, scale = b1)

    def unbias_est(lda, rklist, lklist, lcdlist):
        """Compute sum of unbiased estimators corresponding to all pairs."""
        count = 1

        for rat in rklist:
            lest = ((lda * (rat - 1)) - log(rat)) / T

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

    def min_max(idx, lda, rklist):
        """Compute unbiased estimator corresponding to min or max pair."""
        
        lest = ((lda * (rklist[idx] - 1)) - log(rklist[idx])) / T
        return lest

    # FOR EACH COMPARISON DIGIT COMPUTE KLD FOR ALL DIGITS
    for C in range(0, 10):
        for D in range(0, 10):

            for i in range(0, U):
                KLDiv[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

            for j in range(0, E):
                eKLDiv[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))
                totalNoiseL = 0

                # OPTION 3A: ADD NOISE IN MIDDLE (B: AT END, AFTER / T ABOVE)
                for k in range(0, R):
                    totalNoiseL = totalNoiseL + (noiseL.sample(sample_shape = (1,)))
            
                # COMPUTE AVERAGE OF R POSSIBLE NOISE TERMS
                avNoiseL = totalNoiseL / R
                eKLDiv[C, D, j] = eKLDiv[C, D, j] + avNoiseL

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

                    # COMPUTE UNBIASED ESTIMATORS WITH LAMBDA 0.5 THEN BINARY SEARCH
                    midSum = unbias_est(mid, rKList, lTwoKList, lTwoCDList)

                    # TOLERANCE BETWEEN BINARY SEARCH LIMITS ALWAYS GETS SMALL ENOUGH
                    while abs(high - low) > 0.00000001:

                        lowSum = unbias_est(low, rKList, lZeroKList, lZeroCDList)
                        highSum = unbias_est(high, rKList, lOneKList, lOneCDList)

                        # REDUCE / INCREASE BINARY SEARCH LIMIT DEPENDING ON ABSOLUTE VALUE
                        if abs(lowSum) < abs(highSum):
                            high = mid
                        else:
                            low = mid

                        # SET NEW MIDPOINT
                        mid = 0.5*abs((high - low))
                        midSum = unbias_est(mid, rKList, lThreeKList, lThreeCDList)

                    print(f"\nmidSum: {midSum}")
                    sumLambda = mid
                    print(f"sumLambda: {sumLambda}")

                    # EXTRACT MIN PAIR BY ABSOLUTE VALUE OF EXACT KL DIVERGENCE
                    absKList = [abs(kl) for kl in KList]
                    minKList = sorted(absKList)
                    minAbs = minKList[0]
                    minIndex = KList.index(minAbs)
                    minPair = CDList[minIndex]
                    MIN_COUNT = 1

                    # IF MIN PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT SMALLEST
                    while minPair not in lTwoCDList:        
                        minAbs = minKList[MIN_COUNT]
                        minIndex = KList.index(minAbs)
                        minPair = CDList[minIndex]
                        MIN_COUNT = MIN_COUNT + 1

                    midMinIndex = lTwoCDList.index(minPair)
                    midMinKL = lTwoKList[midMinIndex]

                    low = 0
                    high = 1
                    mid = 0.5

                    # FIND OPTIMAL LAMBDA FOR MIN PAIR
                    while abs(high - low) > 0.00000001:
                
                        lowMinIndex = lZeroCDList.index(minPair)
                        lowMinKL = lZeroKList[lowMinIndex]
                        highMinIndex = lOneCDList.index(minPair)
                        highMinKL = lOneKList[highMinIndex]

                        if abs(lowMinKL) < abs(highMinKL):
                            high = mid
                        else:
                            low = mid

                        mid = 0.5*abs((high - low))
                        midMinKL = min_max(midMinIndex, mid, rKList)

                    print(f"\nmidMinKL: {midMinKL}")
                    minLambda = mid
                    print(f"minLambda: {minLambda}")

                    # EXTRACT MAX PAIR BY REVERSING EXACT KL DIVERGENCE LIST
                    maxKList = sorted(absKList, reverse = True)
                    maxAbs = maxKList[0]
                    maxIndex = KList.index(maxAbs)
                    maxPair = CDList[maxIndex]
                    MAX_COUNT = 1

                    # IF MAX PAIR IS NOT IN LAMBDA 0.5 LIST THEN GET NEXT LARGEST
                    while maxPair not in lTwoCDList:        
                        maxAbs = maxKList[MAX_COUNT]
                        maxIndex = KList.index(maxAbs)
                        maxPair = CDList[maxIndex]
                        MAX_COUNT = MAX_COUNT + 1

                    midMaxIndex = lTwoCDList.index(maxPair)
                    midMaxKL = lTwoKList[midMaxIndex]

                    low = 0
                    high = 1
                    mid = 0.5

                    # FIND OPTIMAL LAMBDA FOR MAX PAIR
                    while abs(high - low) > 0.00000001:

                        lowMaxIndex = lZeroCDList.index(maxPair)
                        lowMaxKL = lZeroKList[lowMaxIndex]
                        highMaxIndex = lOneCDList.index(maxPair)
                        highMaxKL = lOneKList[highMaxIndex]
                
                        if abs(lowMaxKL) < abs(highMaxKL):
                            high = mid
                        else:
                            low = mid

                        mid = 0.5*(abs(high - low))
                        maxLambda = min_max(midMaxIndex, mid, rKList)

                    print(f"\nmidMaxKL: {midMaxKL}")
                    maxLambda = mid
                    print(f"maxLambda: {maxLambda}")

    # CREATE ORDERED DICTIONARIES OF STORED KLD AND DIGITS
    KLDict = dict(zip(KList, CDList))
    orderedKLDict = OrderedDict(sorted(KLDict.items()))
    datafile = open("emnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
    datafile.write("EMNIST: Exact KL Divergence In Order\n")
    datafile.write("Smaller corresponds to more similar digits\n\n")

    eKLDict = dict(zip(eKList, eCDList))
    eOrderedKLDict = OrderedDict(sorted(eKLDict.items()))
    estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_est.txt", "w", encoding = 'utf-8')
    estfile.write("EMNIST: Estimated KL Divergence In Order\n")
    estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    estfile.write("Smaller corresponds to more similar digits\n\n")

    rKLDict = dict(zip(rKList, rCDList))
    rOrderedKLDict = OrderedDict(sorted(rKLDict.items()))
    ratiofile = open(f"em_kld_mid_lap_noise_eps_{eps}_ratio.txt", "w", encoding = 'utf-8')
    ratiofile.write("EMNIST: Ratio Between Exact KL Divergence And Estimator\n")
    ratiofile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    ratiofile.write("Closer to 1 corresponds to a better estimate\n\n")

    # CHECK WHETHER RANKING IS PRESERVED WHEN ESTIMATOR IS USED
    DATA_ROWS = 90
    TOP_COUNT = 0
    BOTTOM_COUNT = 0

    # LOOK AT TOP AND BOTTOM 10% OF DIGIT PAIRS IN EXACT KLD RANKING LIST
    topKLDict = list(orderedKLDict.values())[0 : int(DATA_ROWS / 10)]
    rTopKLDict = list(rOrderedKLDict.values())[0 : int(DATA_ROWS / 2)]
    bottomKLDict = list(orderedKLDict.values())[int(9*(DATA_ROWS / 10)) : DATA_ROWS]
    rBottomKLDict = list(rOrderedKLDict.values())[int(DATA_ROWS / 2) : DATA_ROWS]

    # DO TOP 10% IN EXACT KLD REMAIN IN TOP HALF OF RATIO?
    for ti in topKLDict:
        for tj in rTopKLDict:    
            if tj == ti:
                TOP_COUNT = TOP_COUNT + 1

    # DO BOTTOM 10% IN EXACT KLD REMAIN IN BOTTOM HALF OF RATIO?
    for bi in bottomKLDict:
        for bj in rBottomKLDict:
            if bj == bi:
                BOTTOM_COUNT = BOTTOM_COUNT + 1

    percTopKLD = 100*(TOP_COUNT / int(DATA_ROWS/10))
    percBottomKLD = 100*(BOTTOM_COUNT / int(DATA_ROWS/10))
    ratiofile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(percTopKLD, 1)}%\n")
    ratiofile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(percBottomKLD, 1)}%\n\n")

    lZeroKLDict = dict(zip(lZeroKList, lZeroCDList))
    lZeroOrderedKLDict = OrderedDict(sorted(lZeroKLDict.items()))
    l0estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l0est.txt", "w", encoding = 'utf-8')
    l0estfile.write("EMNIST: Unbiased Estimator Lambda = 0\n")
    l0estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    l0estfile.write(f"Sum: {sum(lZeroKList)}\n\n")

    lOneKLDict = dict(zip(lOneKList, lOneCDList))
    lOneOrderedKLDict = OrderedDict(sorted(lOneKLDict.items()))
    l1estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l1est.txt", "w", encoding = 'utf-8')
    l1estfile.write("EMNIST: Unbiased Estimator Lambda = 1\n")
    l1estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    l1estfile.write(f"Sum: {sum(lOneKList)}\n\n")

    lTwoKLDict = dict(zip(lTwoKList, lTwoCDList))
    lTwoOrderedKLDict = OrderedDict(sorted(lTwoKLDict.items()))
    l2estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l2est.txt", "w", encoding = 'utf-8')
    l2estfile.write("EMNIST: Unbiased Estimator Lambda = 0.5\n")
    l2estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    l2estfile.write(f"Sum: {sum(lTwoKList)}\n\n")

    lThreeKLDict = dict(zip(lThreeKList, lThreeCDList))
    lThreeOrderedKLDict = OrderedDict(sorted(lThreeKLDict.items()))
    l3estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l3est.txt", "w", encoding = 'utf-8')
    l3estfile.write("EMNIST: Unbiased Estimator Optimal Lambda for Sum\n")
    l3estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    l3estfile.write(f"Optimal Lambda {sumLambda} for Sum {midSum}\n\n")

    l4estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l4est.txt", "w", encoding = 'utf-8')
    l4estfile.write("EMNIST: Unbiased Estimator Optimal Lambda Min Pair (7, 1)\n")
    l4estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    l4estfile.write(f"Optimal Lambda {minLambda} for Estimate {midMinKL}\n\n")

    l5estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l5est.txt", "w", encoding = 'utf-8')
    l5estfile.write("EMNIST: Unbiased Estimator Optimal Lambda Max Pair (6, 9)\n")
    l5estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    l5estfile.write(f"Optimal Lambda {maxLambda} for Estimate {midMaxKL}\n\n")

    for i in orderedKLDict:
        datafile.write(f"{i} : {orderedKLDict[i]}\n")

    for j in eOrderedKLDict:
        estfile.write(f"{j} : {eOrderedKLDict[j]}\n")

    for k in rOrderedKLDict:
        ratiofile.write(f"{k} : {rOrderedKLDict[k]}\n")

    for l in lZeroOrderedKLDict:
        l0estfile.write(f"{l} : {lZeroOrderedKLDict[l]}\n")

    for m in lOneOrderedKLDict:
        l1estfile.write(f"{m} : {lOneOrderedKLDict[m]}\n")

    for n in lTwoOrderedKLDict:
        l2estfile.write(f"{n} : {lTwoOrderedKLDict[n]}\n")

    for o in lThreeOrderedKLDict:
        l3estfile.write(f"{o} : {lThreeOrderedKLDict[o]}\n")

    # SHOW ALL RANDOM IMAGES AT THE SAME TIME
    fig, ax = plt.subplots(2, 5, sharex = True, sharey = True)

PLOT_COUNT = 0
for row in ax:
    for col in row:
        randomNumber = random.randint(0, 1399)
        col.imshow(sampleImSet[PLOT_COUNT, randomNumber], cmap = 'gray')
        col.set_title(f'Digit: {PLOT_COUNT}')
        PLOT_COUNT = PLOT_COUNT + 1

plt.ion()
plt.show()
plt.pause(0.001)
input("\nPress [enter] to continue.")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
