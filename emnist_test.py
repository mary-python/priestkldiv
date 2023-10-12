"""Modules provide various time-related functions, generate pseudo-random numbers,
remember the order in which items are added, have cool visual feedback of the current throughput,
create static, animated, and interactive visualisations, provide functionality to automatically
download and cache the EMNIST dataset, and work with arrays in Python."""
import time
import random
from collections import OrderedDict
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
import numpy as np
np.set_printoptions(suppress=True)
np.seterr(invalid='ignore')

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter()
print("\nStarting...")
random.seed(3249583)

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

# STORES FOR ACCURACY OF ESTIMATED KLD
diffKList = []
diffCDList = []
percKList = []
percCDList = []

print("Computing KL divergence...")

# FOR EACH COMPARISON DIGIT COMPUTE KLD FOR ALL DIGITS
for C in range(0, 10):
    for D in range(0, 10):
        for i in range(0, U):
            KLDiv[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

        for j in range(0, E):
            eKLDiv[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

        # ELIMINATE ALL ZERO VALUES WHEN DIGITS ARE IDENTICAL
        if sum(KLDiv[C, D]) != 0.0:
            KList.append(sum(KLDiv[C, D]))
            CDList.append((C, D))

        if sum(eKLDiv[C, D]) != 0.0:
            eKList.append(sum(eKLDiv[C, D]))
            eCDList.append((C, D))

        # COMPUTE ABSOLUTE AND PERCENTAGE MEASURES OF ACCURACY
        absDifference = abs(sum(eKLDiv[C, D]) - sum(KLDiv[C, D]))
        percDifference = 100*abs(sum(eKLDiv[C, D]) / sum(KLDiv[C, D]))

        if absDifference != 0.0:
            diffKList.append(absDifference)
            diffCDList.append((C, D))

        if percDifference != 0.0 and sum(KLDiv[C, D]) != 0.0:
            percKList.append(percDifference)
            percCDList.append((C, D))

# CREATE ORDERED DICTIONARIES OF STORED KLD AND DIGITS
KLDict = dict(zip(KList, CDList))
orderedKLDict = OrderedDict(sorted(KLDict.items()))
datafile = open("emnist_exact_kld_in_order.txt", "w", encoding = 'utf-8')
datafile.write("EMNIST: Exact KL Divergence In Order\n")
datafile.write("Smaller corresponds to more similar digits\n\n")

eKLDict = dict(zip(eKList, eCDList))
eOrderedKLDict = OrderedDict(sorted(eKLDict.items()))
estfile = open("emnist_est_kld_in_order.txt", "w", encoding = 'utf-8')
estfile.write("EMNIST: Estimated KL Divergence In Order\n")
estfile.write("Smaller corresponds to more similar digits\n\n")

diffKLDict = dict(zip(diffKList, diffCDList))
diffOrderedKLDict = OrderedDict(sorted(diffKLDict.items()))
difffile = open("emnist_abs_accuracy_kld.txt", "w", encoding = 'utf-8')
difffile.write("EMNIST: Absolute Accuracy of KL Divergence Estimator\n")
difffile.write("Smaller corresponds to a better estimate\n\n")

percKLDict = dict(zip(percKList, percCDList))
percOrderedKLDict = OrderedDict(sorted(percKLDict.items()))
percfile = open("emnist_perc_accuracy_kld.txt", "w", encoding = 'utf-8')
percfile.write("EMNIST: Percentage Accuracy of KL Divergence Estimator\n")
percfile.write("Closer to 100% corresponds to a better estimate\n\n")

for i in orderedKLDict:
    datafile.write(f"{i} : {orderedKLDict[i]}\n")

for j in eOrderedKLDict:
    estfile.write(f"{j} : {eOrderedKLDict[j]}\n")

for k in diffOrderedKLDict:
    difffile.write(f"{k} : {diffOrderedKLDict[k]}\n")

for l in percOrderedKLDict:
    percfile.write(f"{l} % : {percOrderedKLDict[l]}\n")

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
