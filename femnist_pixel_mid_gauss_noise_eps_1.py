"""Modules provide various time-related functions, compute the natural logarithm of a number, 
remember the order in which items are added, provide both a high- and low-level interface to
the HDF5 library, work with arrays, and carry out fast numerical computations in Python."""
import time
from math import log
from collections import OrderedDict
import matplotlib.pyplot as plt
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

# fetch hdf5 file from current directory
PATH = './data/write_all.hdf5'
file = h5py.File(PATH, 'r')

# create list storing images and labels of each writer
writers = sorted(file.keys())
numWriters = len(writers)
numSampledWriters = int(numWriters / 20)

# randomly sample 5% of writers without replacement
sampledWriters = np.random.choice(numWriters, numSampledWriters, replace = False)
totalDigits = np.zeros(10, dtype = int)

# add up how many times each digit is featured
print("Preprocessing images...")
for i in sampledWriters:
    tempDataset = file[writers[i]]

    for pic in range(len(tempDataset['labels'])):

        for count in range(10):
            if tempDataset['labels'][pic] == count:
                totalDigits[count] = totalDigits[count] + 1

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
digCount = np.zeros(10, dtype = int)

def add_digit(dset):
    """Method to add digit to set corresponding to label."""
    dset[digCount[label]] = smallPic

for i in sampledWriters:

    tempDataset = file[writers[i]]
    PIC_COUNT = 0

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

        label = tempDataset['labels'][PIC_COUNT]

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

        digCount[label] = digCount[label] + 1
        PIC_COUNT = PIC_COUNT + 1

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
uTotalSet = np.ones((1124, 4, 4), dtype = int)
TOTAL_COUNT = 0

def total_set(uset, tset, tcount):
    """Method to add each of the unique images for each digit."""
    for im in uset:
        tset[tcount] = im
        tcount = tcount + 1
    return tcount

TOTAL_COUNT = total_set(uZeroSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uOneSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uTwoSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uThreeSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uFourSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uFiveSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uSixSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uSevenSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uEightSet, uTotalSet, TOTAL_COUNT)
TOTAL_COUNT = total_set(uNineSet, uTotalSet, TOTAL_COUNT)

uTotalSet = unique_images(10, uTotalSet)

# domain for each digit distribution is number of unique images
U = 338

# find and store frequencies of unique images for each digit
uImageSet = np.ones((10, U, 4, 4))
uFreqSet = np.zeros((10, U))
uProbsSet = np.zeros((10, U))
T1 = 11*numSampledWriters # change this term so probs add up to 1

print("Creating probability distributions...")

# smoothing parameter: 0.1 and 1 are too large
ALPHA = 0.01

def smoothed_prob(dset, dig, im, ucount):
    """Method to compute frequencies of unique images and return smoothed probabilities."""
    where = np.where(np.all(im == dset, axis = (1, 2)))
    freq = len(where[0])
    uImageSet[dig, ucount] = im
    uFreqSet[dig, ucount] = int(freq)
    uProbsSet[dig, ucount] = float((freq + ALPHA)/(T1 + (ALPHA*(digCount[dig]))))

for D in range(0, 10):
    UNIQUE_COUNT = 0

    # store image and smoothed probability as well as frequency
    for image in uTotalSet:
        if D == 0:
            smoothed_prob(zeroSet, 0, image, UNIQUE_COUNT)
        elif D == 1:
            smoothed_prob(oneSet, 1, image, UNIQUE_COUNT)
        elif D == 2:
            smoothed_prob(twoSet, 2, image, UNIQUE_COUNT)
        elif D == 3:
            smoothed_prob(threeSet, 3, image, UNIQUE_COUNT)
        elif D == 4:
            smoothed_prob(fourSet, 4, image, UNIQUE_COUNT)
        elif D == 5:
            smoothed_prob(fiveSet, 5, image, UNIQUE_COUNT)
        elif D == 6:
            smoothed_prob(sixSet, 6, image, UNIQUE_COUNT)
        elif D == 7:
            smoothed_prob(sevenSet, 7, image, UNIQUE_COUNT)
        elif D == 8:
            smoothed_prob(eightSet, 8, image, UNIQUE_COUNT)
        elif D == 9:
            smoothed_prob(nineSet, 9, image, UNIQUE_COUNT)

        UNIQUE_COUNT = UNIQUE_COUNT + 1

# for k3 estimator (Schulman) take a small sample of unique images
E = 17

# store images, frequencies and probabilities for this subset
eImageSet = np.ones((10, E, 4, 4))
eFreqSet = np.zeros((10, E))
eProbsSet = np.zeros((10, E))
eTotalFreq = np.zeros(10)

uSampledSet = np.random.choice(U, E, replace = False)
T2 = (11/3)*numSampledWriters*(E/U) # change this term so probs add up to 1

# borrow data from corresponding indices of main image and frequency sets
for D in range(0, 10):
    for i in range(E):
        eImageSet[D, i] = uImageSet[D, uSampledSet[i]]
        eFreqSet[D, i] = uFreqSet[D, uSampledSet[i]]
        eTotalFreq[D] = sum(eFreqSet[D])
        eProbsSet[D, i] = float((eFreqSet[D, i] + ALPHA)/(T2 + (ALPHA*(eTotalFreq[D]))))

# stores for exact KLD
KLDiv = np.zeros((10, 10, U))
sumKLDiv = np.zeros((10, 10))
KList = []
CDList = []

# stores for estimated KLD
eKLDiv = np.zeros((10, 10, E))
eSumKLDiv = np.zeros((10, 10))
eKList = []
eCDList = []

# stores for ratio between exact and estimated KLD
rKList = []
rCDList = []

# stores for unbiased estimate of KLD
lZeroKList = []
lZeroCDList = []
lOneKList = []
lOneCDList = []
lHalfKList = []
lHalfCDList = []
lTenthKList = []
lTenthCDList = []
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

# parameters for the addition of Laplace and Gaussian noise
EPS = 1 # set EPS to be a range of values (0.01 to 4)
DTA = 0.1
A = 0
R = 10

# option 1a: querying entire distribution (b: Monte Carlo sampling)
b1 = log(2) / EPS
b2 = (2*((log(1.25))/DTA)*b1) / EPS

# option 2b: add Gaussian noise (a: Laplace noise)
noiseG = tfp.distributions.Normal(loc = A, scale = b2)

print("Computing KL divergence...")

def unbias_est(lda, rat, lklist, lcdlist, c, d):
    """Compute unbiased estimator using computed ratio."""
    lest = ((lda * (rat - 1)) - log(rat)) / T1

    if lest != 0.0:
        lklist.append(lest)
        lcdlist.append((c, d))

# for each comparison digit compute KLD for all digits
for C in range(0, 10):
    for D in range(0, 10):
        for i in range(0, U):
            KLDiv[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

        for j in range(0, E):
            eKLDiv[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))
            totalNoiseG = 0

            # option 3a: add noise in middle (b: at end, after / T1 above)
            for k in range(0, R):
                totalNoiseG = totalNoiseG + (noiseG.sample(sample_shape = (1,)))
            
            # compute average of R possible noise terms
            avNoiseG = totalNoiseG / R

            eKLDiv[C, D, j] = eKLDiv[C, D, j] + avNoiseG

        # eliminate all zero values when digits are identical
        if sum(KLDiv[C, D]) != 0.0:
            KList.append(sum(KLDiv[C,D]))
            CDList.append((C, D))

        # still need below condition to avoid zero error in ratio
        if sum(eKLDiv[C, D]) != 0.0:
            eKList.append(sum(eKLDiv[C, D]))
            eCDList.append((C, D))

        # compute ratio between exact and estimated KLD
        ratio = abs(sum(eKLDiv[C, D]) / sum(KLDiv[C, D]))

        # eliminate all divide by zero errors
        if ratio != 0.0 and sum(KLDiv[C, D]) != 0.0:
            rKList.append(ratio)
            rCDList.append((C, D))

            # compute unbiased estimators with lambda equal to 0, 1, 0.5, 0.1 etc
            unbias_est(0, ratio, lZeroKList, lZeroCDList, C, D)
            unbias_est(1, ratio, lOneKList, lOneCDList, C, D)
            unbias_est(0.5, ratio, lHalfKList, lHalfCDList, C, D)
            unbias_est(0.1, ratio, lTenthKList, lTenthCDList, C, D)
            unbias_est(0.0875, ratio, lNextKList, lNextCDList, C, D)
            unbias_est(0.0872, ratio, lHighKList, lHighCDList, C, D)
            unbias_est(0.0871, ratio, lLowKList, lLowCDList, C, D)
            unbias_est(0.08715, ratio, lDecideKList, lDecideCDList, C, D)

            # explore lambdas below 0
            unbias_est(-1, ratio, lMinusKList, lMinusCDList, C, D)
            unbias_est(-3, ratio, lTripleKList, lTripleCDList, C, D)

            # find optimal lambdas for min (7, 9) and max (1, 5) pairs
            unbias_est(1.3992, ratio, lMinKList, lMinCDList, C, D)
            unbias_est(0.4633, ratio, lMaxKList, lMaxCDList, C, D)

            # look at extreme lambdas
            unbias_est(10000, ratio, lPosKList, lPosCDList, C, D)
            unbias_est(-10000, ratio, lNegKList, lNegCDList, C, D)

# create ordered dictionaries of stored KLD and digits
KLDict = dict(zip(KList, CDList))
orderedKLDict = OrderedDict(sorted(KLDict.items()))
datafile = open("femnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
datafile.write("FEMNIST: Exact KL Divergence In Order\n")
datafile.write("Smaller corresponds to more similar digits\n\n")

eKLDict = dict(zip(eKList, eCDList))
eOrderedKLDict = OrderedDict(sorted(eKLDict.items()))
estfile = open("fem_est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
estfile.write("FEMNIST: Estimated KL Divergence In Order\n")
estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
estfile.write("Smaller corresponds to more similar digits\n\n")

rKLDict = dict(zip(rKList, rCDList))
rOrderedKLDict = OrderedDict(sorted(rKLDict.items()))
ratiofile = open("fem_ratio_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
ratiofile.write("FEMNIST: Ratio Between Exact KL Divergence And Estimator\n")
ratiofile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
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
l0estfile = open("fem_l0est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l0estfile.write("FEMNIST: Unbiased Estimator Lambda Zero\n")
l0estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l0estfile.write(f"Sum: {sum(lZeroKList)}\n\n")

lOneKLDict = dict(zip(lOneKList, lOneCDList))
lOneOrderedKLDict = OrderedDict(sorted(lOneKLDict.items()))
l1estfile = open("fem_l1est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l1estfile.write("FEMNIST: Unbiased Estimator Lambda One\n")
l1estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l1estfile.write(f"Sum: {sum(lOneKList)}\n\n")

lHalfKLDict = dict(zip(lHalfKList, lHalfCDList))
lHalfOrderedKLDict = OrderedDict(sorted(lHalfKLDict.items()))
l2estfile = open("fem_l2est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l2estfile.write("FEMNIST: Unbiased Estimator Lambda Half\n")
l2estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l2estfile.write(f"Sum: {sum(lHalfKList)}\n\n")

lTenthKLDict = dict(zip(lTenthKList, lTenthCDList))
lTenthOrderedKLDict = OrderedDict(sorted(lTenthKLDict.items()))
l3estfile = open("fem_l3est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l3estfile.write("FEMNIST: Unbiased Estimator Lambda Tenth\n")
l3estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l3estfile.write(f"Sum: {sum(lTenthKList)}\n\n")

lNextKLDict = dict(zip(lNextKList, lNextCDList))
lNextOrderedKLDict = OrderedDict(sorted(lNextKLDict.items()))
l4estfile = open("fem_l4est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l4estfile.write("FEMNIST: Unbiased Estimator Lambda Next\n")
l4estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l4estfile.write(f"Sum: {sum(lNextKList)}\n\n")

lHighKLDict = dict(zip(lHighKList, lHighCDList))
lHighOrderedKLDict = OrderedDict(sorted(lHighKLDict.items()))
l5estfile = open("fem_l5est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l5estfile.write("FEMNIST: Unbiased Estimator Lambda High\n")
l5estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l5estfile.write(f"Sum: {sum(lHighKList)}\n\n")

lLowKLDict = dict(zip(lLowKList, lLowCDList))
lLowOrderedKLDict = OrderedDict(sorted(lLowKLDict.items()))
l6estfile = open("fem_l6est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l6estfile.write("FEMNIST: Unbiased Estimator Lambda Low\n")
l6estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l6estfile.write(f"Sum: {sum(lLowKList)}\n\n")
lDecideKLDict = dict(zip(lDecideKList, lDecideCDList))
lDecideOrderedKLDict = OrderedDict(sorted(lDecideKLDict.items()))
l7estfile = open("fem_l7est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l7estfile.write("FEMNIST: Unbiased Estimator Lambda Decide\n")
l7estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l7estfile.write(f"Sum: {sum(lDecideKList)}\n\n")

lMinusKLDict = dict(zip(lMinusKList, lMinusCDList))
lMinusOrderedKLDict = OrderedDict(sorted(lMinusKLDict.items()))
l8estfile = open("fem_l8est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l8estfile.write("FEMNIST: Unbiased Estimator Lambda Minus\n")
l8estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l8estfile.write(f"Sum: {sum(lMinusKList)}\n\n")

lTripleKLDict = dict(zip(lTripleKList, lTripleCDList))
lTripleOrderedKLDict = OrderedDict(sorted(lTripleKLDict.items()))
l9estfile = open("fem_l9est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l9estfile.write("FEMNIST: Unbiased Estimator Lambda Triple\n")
l9estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l9estfile.write(f"Sum: {sum(lTripleKList)}\n\n")

lMinKLDict = dict(zip(lMinKList, lMinCDList))
lMinOrderedKLDict = OrderedDict(sorted(lMinKLDict.items()))
l10estfile = open("fem_l10est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l10estfile.write("FEMNIST: Unbiased Estimator Lambda Min\n")
l10estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l10estfile.write(f"Sum: {sum(lMinKList)}\n\n")

lMaxKLDict = dict(zip(lMaxKList, lMaxCDList))
lMaxOrderedKLDict = OrderedDict(sorted(lMaxKLDict.items()))
l11estfile = open("fem_l11est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l11estfile.write("FEMNIST: Unbiased Estimator Lambda Max\n")
l11estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l11estfile.write(f"Sum: {sum(lMaxKList)}\n\n")

lPosKLDict = dict(zip(lPosKList, lPosCDList))
lPosOrderedKLDict = OrderedDict(sorted(lPosKLDict.items()))
l12estfile = open("fem_l12est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l12estfile.write("FEMNIST: Unbiased Estimator Lambda Pos\n")
l12estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
l12estfile.write(f"Sum: {sum(lPosKList)}\n\n")

lNegKLDict = dict(zip(lNegKList, lNegCDList))
lNegOrderedKLDict = OrderedDict(sorted(lNegKLDict.items()))
l13estfile = open("fem_l13est_kld_mid_gauss_noise_eps_1.txt", "w", encoding = 'utf-8')
l13estfile.write("FEMNIST: Unbiased Estimator Lambda Neg\n")
l13estfile.write("Gaussian Noise in Middle, no Monte Carlo, Eps = 1\n")
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

for n in lHalfOrderedKLDict:
    l2estfile.write(f"{n} : {lHalfOrderedKLDict[n]}\n")

for o in lTenthOrderedKLDict:
    l3estfile.write(f"{o} : {lTenthOrderedKLDict[o]}\n")

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

# plot a random example of each digit
fig, axs = plt.subplots(3, 4, figsize = (8, 7))

for i in range(3):
    for j in range(4):
        axs[i, j].axis('off')

# check whether images and labels match
zeroIndex = np.random.randint(len(zeroSet))
axs[0, 0].imshow(zeroSet[zeroIndex])
axs[0, 0].title.set_text("0")

oneIndex = np.random.randint(len(oneSet))
axs[0, 1].imshow(oneSet[oneIndex])
axs[0, 1].title.set_text("1")

twoIndex = np.random.randint(len(twoSet))
axs[0, 2].imshow(twoSet[twoIndex])
axs[0, 2].title.set_text("2")

threeIndex = np.random.randint(len(threeSet))
axs[0, 3].imshow(threeSet[threeIndex])
axs[0, 3].title.set_text("3")

fourIndex = np.random.randint(len(fourSet))
axs[1, 0].imshow(fourSet[fourIndex])
axs[1, 0].title.set_text("4")

fiveIndex = np.random.randint(len(fiveSet))
axs[1, 1].imshow(fiveSet[fiveIndex])
axs[1, 1].title.set_text("5")

sixIndex = np.random.randint(len(sixSet))
axs[1, 2].imshow(sixSet[sixIndex])
axs[1, 2].title.set_text("6")

sevenIndex = np.random.randint(len(sevenSet))
axs[1, 3].imshow(sevenSet[sevenIndex])
axs[1, 3].title.set_text("7")

eightIndex = np.random.randint(len(eightSet))
axs[2, 1].imshow(eightSet[eightIndex])
axs[2, 1].title.set_text("8")

nineIndex = np.random.randint(len(nineSet))
axs[2, 2].imshow(nineSet[nineIndex])
axs[2, 2].title.set_text("9")

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
