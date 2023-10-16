"""Modules provide various time-related functions, compute the natural logarithm of a number, 
remember the order in which items are added, provide both a high- and low-level interface to
the HDF5 library, and work with arrays in Python."""
import time
from math import log
from collections import OrderedDict
import matplotlib.pyplot as plt
import h5py
import numpy as np
np.set_printoptions(suppress=True)
np.seterr(invalid='ignore')

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

        for d in range(10):
            if tempDataset['labels'][pic] == d:
                totalDigits[d] = totalDigits[d] + 1

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

print("Creating probability distributions...")

# smoothing parameter: 0.1 and 1 are too large
ALPHA = 0.01

def smoothed_prob(dset, dig, im, ucount):
    """Method to compute frequencies of unique images and return smoothed probabilities."""
    where = np.where(np.all(im == dset, axis = (1, 2)))
    freq = len(where[0])
    uImageSet[dig, ucount] = im
    uFreqSet[dig, ucount] = int(freq)
    uProbsSet[dig, ucount] = float((freq + ALPHA)/(11*numSampledWriters + (ALPHA*(digCount[dig]))))

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
T = (11/3)*numSampledWriters*(E/U)

# borrow data from corresponding indices of main image and frequency sets
for D in range(0, 10):
    for i in range(E):
        eImageSet[D, i] = uImageSet[D, uSampledSet[i]]
        eFreqSet[D, i] = uFreqSet[D, uSampledSet[i]]
        eTotalFreq[D] = sum(eFreqSet[D])
        eProbsSet[D, i] = float((eFreqSet[D, i] + ALPHA)/(T + (ALPHA*(eTotalFreq[D]))))

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
LAMBDA_ZERO = 0
LAMBDA_ONE = 1

print("Computing KL divergence...")

# for each comparison digit compute KLD for all digits
for C in range(0, 10):
    for D in range(0, 10):
        for i in range(0, U):
            KLDiv[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

        for j in range(0, E):
            eKLDiv[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

        # eliminate all zero values when digits are identical
        if sum(KLDiv[C, D]) != 0.0:
            KList.append(sum(KLDiv[C,D]))
            CDList.append((C, D))

        if sum(eKLDiv[C, D]) != 0.0:
            eKList.append(sum(eKLDiv[C, D]))
            eCDList.append((C, D))

        # compute ratio between exact and estimated KLD
        ratio = abs(sum(eKLDiv[C, D]) / sum(KLDiv[C, D]))

        # eliminate all divide by zero errors
        if ratio != 0.0 and sum(KLDiv[C, D]) != 0.0:
            rKList.append(ratio)
            rCDList.append((C, D))

            # compute unbiased estimators with lambda equal to zero and one respectively
            lZeroEst = ((LAMBDA_ZERO * (ratio - 1)) - log(ratio)) / (11*numSampledWriters)
            lOneEst = ((LAMBDA_ONE * (ratio - 1)) - log(ratio)) / (11*numSampledWriters)

            if lZeroEst != 0.0:
                lZeroKList.append(lZeroEst)
                lZeroCDList.append((C, D))

            if lOneEst != 0.0:
                lOneKList.append(lOneEst)
                lOneCDList.append((C, D))

# create ordered dictionaries of stored KLD and digits
KLDict = dict(zip(KList, CDList))
orderedKLDict = OrderedDict(sorted(KLDict.items()))
datafile = open("femnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
datafile.write("FEMNIST: Exact KL Divergence In Order\n")
datafile.write("Smaller corresponds to more similar digits\n\n")

eKLDict = dict(zip(eKList, eCDList))
eOrderedKLDict = OrderedDict(sorted(eKLDict.items()))
estfile = open("femnist_est_kld_in_order.txt", "w", encoding = 'utf-8')
estfile.write("FEMNIST: Estimated KL Divergence In Order\n")
estfile.write("Smaller corresponds to more similar digits\n\n")

rKLDict = dict(zip(rKList, rCDList))
rOrderedKLDict = OrderedDict(sorted(rKLDict.items()))
ratiofile = open("femnist_ratio_kld_in_order.txt", "w", encoding = 'utf-8')
ratiofile.write("FEMNIST: Ratio Between Exact KL Divergence And Estimator\n")
ratiofile.write("Closer to 1 corresponds to a better estimate\n\n")

lZeroKLDict = dict(zip(lZeroKList, lZeroCDList))
lZeroOrderedKLDict = OrderedDict(sorted(lZeroKLDict.items()))
l0estfile = open("femnist_l0est_kld_in_order.txt", "w", encoding = 'utf-8')
l0estfile.write("FEMNIST: Unbiased Estimator Lambda Zero\n")
l0estfile.write(f"Sum: {sum(lZeroKList)}\n\n")

lOneKLDict = dict(zip(lOneKList, lOneCDList))
lOneOrderedKLDict = OrderedDict(sorted(lOneKLDict.items()))
l1estfile = open("femnist_l1est_kld_in_order.txt", "w", encoding = 'utf-8')
l1estfile.write("FEMNIST: Unbiased Estimator Lambda One\n")
l1estfile.write(f"Sum: {sum(lOneKList)}\n\n")

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
