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
minSum = np.zeros(12)
minPairEst = np.zeros(12)
maxPairEst = np.zeros(12)
sumLambda = np.zeros(12)
minPairLambda = np.zeros(12)
maxPairLambda = np.zeros(12)

# STORES FOR RANKING PRESERVATION ANALYSIS
rPercTopKLD = np.zeros(12)
rPercBottomKLD = np.zeros(12)
sumPercTopKLD = np.zeros(12)
sumPercBottomKLD = np.zeros(12)
minPercTopKLD = np.zeros(12)
minPercBottomKLD = np.zeros(12)
maxPercTopKLD = np.zeros(12)
maxPercBottomKLD = np.zeros(12)

INDEX_COUNT = 0

epsset = [0.001, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]

for eps in epsset:

    print(f"Computing KL divergence for epsilon = {eps}...")

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
    
    def min_max(lda, rklist, lklist, lcdlist, mp):
        """Compute unbiased estimator corresponding to min or max pair."""
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

        lmi = lcdlist.index(mp)
        return lklist[lmi]

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

                    # COMPUTE UNBIASED ESTIMATORS WITH LAMBDA 0, 1, 0.5 THEN BINARY SEARCH
                    lowSum = unbias_est(low, rKList, zeroKList, zeroCDList)
                    highSum = unbias_est(high, rKList, oneKList, oneCDList)
                    minSum[INDEX_COUNT] = unbias_est(mid, rKList, halfKList, halfCDList)

                    # TOLERANCE BETWEEN BINARY SEARCH LIMITS ALWAYS GETS SMALL ENOUGH
                    while abs(high - low) > 0.00000001:

                        lowKList = []
                        lowCDList = []
                        highKList = []
                        highCDList = []
                        sumKList = []
                        sumCDList = []

                        lowSum = unbias_est(low, rKList, lowKList, lowCDList)
                        highSum = unbias_est(high, rKList, highKList, highCDList)

                        # REDUCE / INCREASE BINARY SEARCH LIMIT DEPENDING ON ABSOLUTE VALUE
                        if abs(lowSum) < abs(highSum):
                            high = mid
                        else:
                            low = mid

                        # SET NEW MIDPOINT
                        mid = (0.5*abs((high - low))) + low
                        minSum[INDEX_COUNT] = unbias_est(mid, rKList, sumKList, sumCDList)

                    sumLambda[INDEX_COUNT] = mid

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
                    minPairEst[INDEX_COUNT] = halfKList[midMinIndex]

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

                        lowMinKL = min_max(low, rKList, lowKList, lowCDList, minPair)
                        highMinKL = min_max(high, rKList, highKList, highCDList, minPair)

                        if abs(lowMinKL) < abs(highMinKL):
                            high = mid
                        else:
                            low = mid

                        mid = (0.5*abs((high - low))) + low
                        minPairEst[INDEX_COUNT] = min_max(mid, rKList, minKList, minCDList, minPair)

                    minPairLambda[INDEX_COUNT] = mid

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
                    maxPairEst[INDEX_COUNT] = halfKList[midMaxIndex]

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

                        lowMaxKL = min_max(low, rKList, lowKList, lowCDList, maxPair)
                        highMaxKL = min_max(high, rKList, highKList, highCDList, maxPair)
                
                        if abs(lowMaxKL) < abs(highMaxKL):
                            high = mid
                        else:
                            low = mid

                        mid = (0.5*(abs(high - low))) + low
                        maxPairEst[INDEX_COUNT] = min_max(mid, rKList, maxKList, maxCDList, maxPair)

                    maxPairLambda[INDEX_COUNT] = mid

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
    estfile = open(f"em_kld_mid_lap_noise_eps_{eps}_est.txt", "w", encoding = 'utf-8')
    estfile.write("EMNIST: Estimated KL Divergence In Order\n")
    estfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    estfile.write("Smaller corresponds to more similar digits\n\n")

    for j in eOrderedKLDict:
        estfile.write(f"{j} : {eOrderedKLDict[j]}\n")

    rKLDict = dict(zip(rKList, rCDList))
    rOrderedKLDict = OrderedDict(sorted(rKLDict.items()))
    ratiofile = open(f"em_kld_mid_lap_noise_eps_{eps}_ratio.txt", "w", encoding = 'utf-8')
    ratiofile.write("EMNIST: Ratio Between Exact KL Divergence And Estimator\n")
    ratiofile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    ratiofile.write("Closer to 1 corresponds to a better estimate\n\n")

    rPercTopKLD[INDEX_COUNT] = rank_pres(0, orderedKLDict, rOrderedKLDict)
    rPercBottomKLD[INDEX_COUNT] = rank_pres(1, orderedKLDict, rOrderedKLDict)
    ratiofile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(rPercTopKLD[INDEX_COUNT], 1)}%\n")
    ratiofile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(rPercBottomKLD[INDEX_COUNT], 1)}%\n\n")

    for k in rOrderedKLDict:
        ratiofile.write(f"{k} : {rOrderedKLDict[k]}\n")

    sumKLDict = dict(zip(sumKList, sumCDList))
    sumOrderedKLDict = OrderedDict(sorted(sumKLDict.items()))
    sumfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l3est.txt", "w", encoding = 'utf-8')
    sumfile.write("EMNIST: Unbiased Estimator Optimal Lambda for Sum\n")
    sumfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    sumfile.write(f"Optimal Lambda {sumLambda[INDEX_COUNT]} for Sum {minSum[INDEX_COUNT]}\n\n")

    sumPercTopKLD[INDEX_COUNT] = rank_pres(0, orderedKLDict, sumOrderedKLDict)
    sumPercBottomKLD[INDEX_COUNT] = rank_pres(1, orderedKLDict, sumOrderedKLDict)
    sumfile.write(f"Top 10% exact KLD -> top half sum ranking: {round(sumPercTopKLD[INDEX_COUNT], 1)}%\n")
    sumfile.write(f"Bottom 10% exact KLD -> bottom half sum ranking: {round(sumPercBottomKLD[INDEX_COUNT], 1)}%\n\n")

    minKLDict = dict(zip(minKList, minCDList))
    minOrderedKLDict = OrderedDict(sorted(minKLDict.items()))
    minfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l4est.txt", "w", encoding = 'utf-8')
    minfile.write("EMNIST: Unbiased Estimator Optimal Lambda Min Pair (7, 1)\n")
    minfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    minfile.write(f"Optimal Lambda {minPairLambda[INDEX_COUNT]} for Estimate {minPairEst[INDEX_COUNT]}\n\n")

    minPercTopKLD[INDEX_COUNT] = rank_pres(0, orderedKLDict, minOrderedKLDict)
    minPercBottomKLD[INDEX_COUNT] = rank_pres(1, orderedKLDict, minOrderedKLDict)
    minfile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(minPercTopKLD[INDEX_COUNT], 1)}%\n")
    minfile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(minPercBottomKLD[INDEX_COUNT], 1)}%\n\n")

    maxKLDict = dict(zip(maxKList, maxCDList))
    maxOrderedKLDict = OrderedDict(sorted(maxKLDict.items()))
    maxfile = open(f"em_kld_mid_lap_noise_eps_{eps}_l5est.txt", "w", encoding = 'utf-8')
    maxfile.write("EMNIST: Unbiased Estimator Optimal Lambda Max Pair (6, 9)\n")
    maxfile.write(f"Laplace Noise in Middle, no Monte Carlo, Eps = {eps}\n")
    maxfile.write(f"Optimal Lambda {maxPairLambda[INDEX_COUNT]} for Estimate {maxPairEst[INDEX_COUNT]}\n\n")

    maxPercTopKLD[INDEX_COUNT] = rank_pres(0, orderedKLDict, maxOrderedKLDict)
    maxPercBottomKLD[INDEX_COUNT] = rank_pres(1, orderedKLDict, maxOrderedKLDict)
    maxfile.write(f"Top 10% exact KLD -> top half ratio ranking: {round(maxPercTopKLD[INDEX_COUNT], 1)}%\n")
    maxfile.write(f"Bottom 10% exact KLD -> bottom half ratio ranking: {round(maxPercBottomKLD[INDEX_COUNT], 1)}%\n\n")

    INDEX_COUNT = INDEX_COUNT + 1

# PLOT LAMBDAS FOR EACH EPSILON
ax1 = plt.axes()
ax1.plot(epsset, sumLambda)
ax1.set_xlabel("Value of epsilon")
ax1.set_ylabel("Lambda to minimise KLD for all pairs")
ax1.set_title("How epsilon affects lambda (all pairs, EMNIST)")
ax1.savefig("Emnist_mid_lap_lambda_sum.png")

ax2 = plt.axes()
ax2.plot(epsset, minPairLambda, color = 'b', label = 'min pair')
ax2.plot(epsset, maxPairLambda, color = 'r', label = 'max pair')
ax2.legend(loc = 'best')
ax2.set_xlabel("Value of epsilon")
ax2.set_ylabel("Lambda to minimise KLD for min or max pair")
ax2.set_title("How epsilon affects lambda (min and max pair, EMNIST)")
ax2.savefig("Emnist_mid_lap_lambda_min_max.png")

# PLOT SUM / ESTIMATES FOR EACH EPSILON
ax3 = plt.axes()
ax3.plot(epsset, minSum)
ax3.set_xlabel("Value of epsilon")
ax3.set_ylabel("Minimal KLD for all pairs")
ax3.set_title("How epsilon affects minimal KLD (all pairs, EMNIST)")
ax3.savefig("Emnist_mid_lap_est_sum.png")

ax4 = plt.axes()
ax4.plot(epsset, minPairEst, color = 'b', label = 'min pair')
ax4.plot(epsset, maxPairEst, color = 'r', label = 'max pair')
ax4.legend(loc = 'best')
ax4.set_xlabel("Value of epsilon")
ax4.set_ylabel("Minimal KLD for min or max pair")
ax4.set_title("How epsilon affects minimal KLD (min and max pair, EMNIST)")
ax4.savefig("Emnist_mid_lap_est_min_max.png")

# PLOT RANKING PRESERVATIONS FOR EACH EPSILON
ax5 = plt.axes()
ax5.plot(epsset, rPercTopKLD, color = 'b', label = 'top 10%/half')
ax5.plot(epsset, rPercBottomKLD, color = 'r', label = 'bottom 10%/half')
ax5.legend(loc = 'best')
ax5.set_xlabel("Value of epsilon")
ax5.set_ylabel(f"Top/bottom 10% staying in top/bottom half")
ax5.set_title("Ranking preservation for ratio between exact and estimated KLD")
ax5.savefig("Emnist_mid_lap_perc_ratio.png")

ax6 = plt.axes()
ax6.plot(epsset, sumPercTopKLD, color = 'b', label = 'top 10%/half')
ax6.plot(epsset, sumPercBottomKLD, color = 'r', label = 'bottom 10%/half')
ax6.legend(loc = 'best')
ax6.set_xlabel("Value of epsilon")
ax6.set_ylabel(f"Top/bottom 10% staying in top/bottom half")
ax6.set_title("Ranking preservation for minimal KLD (all pairs, EMNIST)")
ax6.savefig("Emnist_mid_lap_perc_sum.png")

ax7 = plt.axes()
ax7.plot(epsset, minPercTopKLD, color = 'b', label = 'top 10%/half')
ax7.plot(epsset, minPercBottomKLD, color = 'r', label = 'bottom 10%/half')
ax7.legend(loc = 'best')
ax7.set_xlabel("Value of epsilon")
ax7.set_ylabel(f"Top/bottom 10% staying in top/bottom half")
ax7.set_title("Ranking preservation for minimal KLD (min pair, EMNIST)")
ax7.savefig("Emnist_mid_lap_perc_min.png")

ax8 = plt.axes()
ax8.plot(epsset, maxPercTopKLD, color = 'b', label = 'top 10%/half')
ax8.plot(epsset, maxPercBottomKLD, color = 'r', label = 'bottom 10%/half')
ax8.legend(loc = 'best')
ax8.set_xlabel("Value of epsilon")
ax8.set_ylabel(f"Top/bottom 10% staying in top/bottom half")
ax8.set_title("Ranking preservation for minimal KLD (max pair, EMNIST)")
ax8.savefig("Emnist_mid_lap_perc_max.png")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
