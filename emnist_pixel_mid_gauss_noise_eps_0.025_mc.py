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
lFiveKList = []
lFiveCDList = []
lSixKList = []
lSixCDList = []
lNextKList = []
lNextCDList = []
lHighKList = []
lHighCDList = []
lLowKList = []
lLowCDList = []
lDecideKList = []
lDecideCDList = []
lMinusKList = []
lMinusCDList = []
lTripleKList = []
lTripleCDList = []
lMinKList = []
lMinCDList = []
lMaxKList = []
lMaxCDList = []
lPosKList = []
lPosCDList = []
lNegKList = []
lNegCDList = []

# PARAMETERS FOR THE ADDITION OF LAPLACE AND GAUSSIAN NOISE
EPS = 0.025 # SET EPS TO BE A RANGE OF VALUES (0.01 TO 4)
DTA = 0.1
A = 0
R = 10

# OPTION 1B: MONTE CARLO SAMPLING (A: QUERYING ENTIRE DISTRIBUTION)
b1 = (1 + log(2)) / EPS
b2 = (2*((log(1.25))/DTA)*b1) / EPS

# OPTION 2B: ADD GAUSSIAN NOISE (A: LAPLACE NOISE)
noiseG = tfp.distributions.Normal(loc = A, scale = b2)

print("Computing KL divergence...")

def unbias_est(lda, rat, lklist, lcdlist, c, d):
    """Compute unbiased estimator using computed ratio."""
    lest = ((lda * (rat - 1)) - log(rat)) / T

    if lest != 0.0:
        lklist.append(lest)
        lcdlist.append((c, d))

# FOR EACH COMPARISON DIGIT COMPUTE KLD FOR ALL DIGITS
for C in range(0, 10):
    for D in range(0, 10):
        for i in range(0, U):
            KLDiv[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

        for j in range(0, E):
            eKLDiv[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))
            totalNoiseG = 0

            # OPTION 3A: ADD NOISE IN MIDDLE (B: AT END, AFTER / T ABOVE)
            for k in range(0, R):
                totalNoiseG = totalNoiseG + (noiseG.sample(sample_shape = (1,)))
            
            # COMPUTE AVERAGE OF R POSSIBLE NOISE TERMS
            avNoiseG = totalNoiseG / R

            eKLDiv[C, D, j] = eKLDiv[C, D, j] + avNoiseG

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

            # COMPUTE UNBIASED ESTIMATORS WITH LAMBDA 0,1 THEN BINARY SEARCH
            unbias_est(0, ratio, lZeroKList, lZeroCDList, C, D)
            unbias_est(1, ratio, lOneKList, lOneCDList, C, D)
            unbias_est(5, ratio, lFiveKList, lFiveCDList, C, D)
            unbias_est(6, ratio, lSixKList, lSixCDList, C, D)
            unbias_est(5.9, ratio, lNextKList, lNextCDList, C, D)
            unbias_est(5.978, ratio, lHighKList, lHighCDList, C, D)
            unbias_est(5.977, ratio, lLowKList, lLowCDList, C, D)
            unbias_est(5.9775, ratio, lDecideKList, lDecideCDList, C, D)

            # EXPLORE LAMBDAS BELOW 0
            unbias_est(-1, ratio, lMinusKList, lMinusCDList, C, D)
            unbias_est(-3, ratio, lTripleKList, lTripleCDList, C, D)

            # FIND OPTIMAL LAMBDAS FOR MIN (0, 5) AND MAX (6, 9) PAIRS
            unbias_est(3.0594, ratio, lMinKList, lMinCDList, C, D)
            unbias_est(7.2723, ratio, lMaxKList, lMaxCDList, C, D)

            # LOOK AT EXTREME LAMBDAS
            unbias_est(10000, ratio, lPosKList, lPosCDList, C, D)
            unbias_est(-10000, ratio, lNegKList, lNegCDList, C, D)

# CREATE ORDERED DICTIONARIES OF STORED KLD AND DIGITS
KLDict = dict(zip(KList, CDList))
orderedKLDict = OrderedDict(sorted(KLDict.items()))
datafile = open("emnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
datafile.write("EMNIST: Exact KL Divergence In Order\n")
datafile.write("Smaller corresponds to more similar digits\n\n")

eKLDict = dict(zip(eKList, eCDList))
eOrderedKLDict = OrderedDict(sorted(eKLDict.items()))
estfile = open("em_est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
estfile.write("EMNIST: Estimated KL Divergence In Order\n")
estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
estfile.write("Smaller corresponds to more similar digits\n\n")

rKLDict = dict(zip(rKList, rCDList))
rOrderedKLDict = OrderedDict(sorted(rKLDict.items()))
ratiofile = open("em_ratio_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
ratiofile.write("EMNIST: Ratio Between Exact KL Divergence And Estimator\n")
ratiofile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
ratiofile.write("Closer to 1 corresponds to a better estimate\n\n")

# check whether ranking is preserved when estimator is used
DATA_ROWS = 90
TOP_COUNT = 0
BOTTOM_COUNT = 0

# look at top and bottom 10% of digit pairs in exact KLD ranking list
topKLDict = list(orderedKLDict.values())[0 : int(DATA_ROWS / 10)]
rTopKLDict = list(rOrderedKLDict.values())[0 : int(DATA_ROWS / 2)]
bottomKLDict = list(orderedKLDict.values())[int(9*(DATA_ROWS / 10)) : DATA_ROWS]
rBottomKLDict = list(rOrderedKLDict.values())[int(DATA_ROWS / 2) : DATA_ROWS]

# do top 10% in exact KLD remain in top half of ratio?
for ti in topKLDict:
    for tj in rTopKLDict:    
        if tj == ti:
            TOP_COUNT = TOP_COUNT + 1

# do bottom 10% in exact KLD remain in bottom half of ratio?
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
l0estfile = open("em_l0est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l0estfile.write("EMNIST: Unbiased Estimator Lambda Zero\n")
l0estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l0estfile.write(f"Sum: {sum(lZeroKList)}\n\n")

lOneKLDict = dict(zip(lOneKList, lOneCDList))
lOneOrderedKLDict = OrderedDict(sorted(lOneKLDict.items()))
l1estfile = open("em_l1est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l1estfile.write("EMNIST: Unbiased Estimator Lambda One\n")
l1estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l1estfile.write(f"Sum: {sum(lOneKList)}\n\n")

lFiveKLDict = dict(zip(lFiveKList, lFiveCDList))
lFiveOrderedKLDict = OrderedDict(sorted(lFiveKLDict.items()))
l2estfile = open("em_l2est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l2estfile.write("EMNIST: Unbiased Estimator Lambda Five\n")
l2estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l2estfile.write(f"Sum: {sum(lFiveKList)}\n\n")

lSixKLDict = dict(zip(lSixKList, lSixCDList))
lSixOrderedKLDict = OrderedDict(sorted(lSixKLDict.items()))
l3estfile = open("em_l3est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l3estfile.write("EMNIST: Unbiased Estimator Lambda Six\n")
l3estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l3estfile.write(f"Sum: {sum(lSixKList)}\n\n")

lNextKLDict = dict(zip(lNextKList, lNextCDList))
lNextOrderedKLDict = OrderedDict(sorted(lNextKLDict.items()))
l4estfile = open("em_l4est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l4estfile.write("EMNIST: Unbiased Estimator Lambda Next\n")
l4estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l4estfile.write(f"Sum: {sum(lNextKList)}\n\n")

lHighKLDict = dict(zip(lHighKList, lHighCDList))
lHighOrderedKLDict = OrderedDict(sorted(lHighKLDict.items()))
l5estfile = open("em_l5est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l5estfile.write("EMNIST: Unbiased Estimator Lambda High\n")
l5estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l5estfile.write(f"Sum: {sum(lHighKList)}\n\n")

lLowKLDict = dict(zip(lLowKList, lLowCDList))
lLowOrderedKLDict = OrderedDict(sorted(lLowKLDict.items()))
l6estfile = open("em_l6est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l6estfile.write("EMNIST: Unbiased Estimator Lambda Low\n")
l6estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l6estfile.write(f"Sum: {sum(lLowKList)}\n\n")

lDecideKLDict = dict(zip(lDecideKList, lDecideCDList))
lDecideOrderedKLDict = OrderedDict(sorted(lDecideKLDict.items()))
l7estfile = open("em_l7est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l7estfile.write("EMNIST: Unbiased Estimator Lambda Decide\n")
l7estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l7estfile.write(f"Sum: {sum(lDecideKList)}\n\n")

lMinusKLDict = dict(zip(lMinusKList, lMinusCDList))
lMinusOrderedKLDict = OrderedDict(sorted(lMinusKLDict.items()))
l8estfile = open("em_l8est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l8estfile.write("EMNIST: Unbiased Estimator Lambda Minus\n")
l8estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l8estfile.write(f"Sum: {sum(lMinusKList)}\n\n")

lTripleKLDict = dict(zip(lTripleKList, lTripleCDList))
lTripleOrderedKLDict = OrderedDict(sorted(lTripleKLDict.items()))
l9estfile = open("em_l9est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l9estfile.write("EMNIST: Unbiased Estimator Lambda Triple\n")
l9estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l9estfile.write(f"Sum: {sum(lTripleKList)}\n\n")

lMinKLDict = dict(zip(lMinKList, lMinCDList))
lMinOrderedKLDict = OrderedDict(sorted(lMinKLDict.items()))
l10estfile = open("em_l10est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l10estfile.write("EMNIST: Unbiased Estimator Lambda Min\n")
l10estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l10estfile.write(f"Sum: {sum(lMinKList)}\n\n")

lMaxKLDict = dict(zip(lMaxKList, lMaxCDList))
lMaxOrderedKLDict = OrderedDict(sorted(lMaxKLDict.items()))
l11estfile = open("em_l11est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l11estfile.write("EMNIST: Unbiased Estimator Lambda Max\n")
l11estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l11estfile.write(f"Sum: {sum(lMaxKList)}\n\n")

lPosKLDict = dict(zip(lPosKList, lPosCDList))
lPosOrderedKLDict = OrderedDict(sorted(lPosKLDict.items()))
l12estfile = open("em_l12est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l12estfile.write("EMNIST: Unbiased Estimator Lambda Pos\n")
l12estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l12estfile.write(f"Sum: {sum(lPosKList)}\n\n")

lNegKLDict = dict(zip(lNegKList, lNegCDList))
lNegOrderedKLDict = OrderedDict(sorted(lNegKLDict.items()))
l13estfile = open("em_l13est_kld_mid_gauss_noise_eps_0.025_mc.txt", "w", encoding = 'utf-8')
l13estfile.write("EMNIST: Unbiased Estimator Lambda Neg\n")
l13estfile.write("Gaussian Noise in Middle, Monte Carlo, Eps = 0.025\n")
l13estfile.write(f"Sum: {sum(lNegKList)}\n\n")

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

for n in lFiveOrderedKLDict:
    l2estfile.write(f"{n} : {lFiveOrderedKLDict[n]}\n")

for o in lSixOrderedKLDict:
    l3estfile.write(f"{o} : {lSixOrderedKLDict[o]}\n")

for p in lNextOrderedKLDict:
    l4estfile.write(f"{p} : {lNextOrderedKLDict[p]}\n")

for q in lHighOrderedKLDict:
    l5estfile.write(f"{q} : {lHighOrderedKLDict[q]}\n")

for r in lLowOrderedKLDict:
    l6estfile.write(f"{r} : {lLowOrderedKLDict[r]}\n")

for s in lDecideOrderedKLDict:
    l7estfile.write(f"{s} : {lDecideOrderedKLDict[s]}\n")

for t in lMinusOrderedKLDict:
    l8estfile.write(f"{t} : {lMinusOrderedKLDict[t]}\n")

for u in lTripleOrderedKLDict:
    l9estfile.write(f"{u} : {lTripleOrderedKLDict[u]}\n")

for v in lMinOrderedKLDict:
    l10estfile.write(f"{v} : {lMinOrderedKLDict[v]}\n")

for w in lMaxOrderedKLDict:
    l11estfile.write(f"{w} : {lMaxOrderedKLDict[w]}\n")

for x in lPosOrderedKLDict:
    l12estfile.write(f"{x} : {lPosOrderedKLDict[x]}\n")

for y in lNegOrderedKLDict:
    l13estfile.write(f"{y} : {lNegOrderedKLDict[y]}\n")

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
