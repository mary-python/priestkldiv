import time
import numpy as np
np.set_printoptions(suppress=True)
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter()
print("\nStarting...")
random.seed(3249583)

# LOAD TRAINING AND TEST SAMPLES FOR 'DIGITS' SUBSET
from emnist import extract_training_samples, extract_test_samples
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
totalCount = 0

# ADD DIGIT TO SET, INDEX TO INDEX SET AND INCREMENT COUNT
def addDigit(dg, pic, set, iset, count, tc):
    set[dg, count[dg]] = pic
    iset[dg, count[dg]] = tc
    count[dg] = count[dg] + 1

# SPLIT NUMBERS 0-9
for digit in labels:
    
    # CALL FUNCTION DEFINED ABOVE
    addDigit(digit, digit, digitSet, digitIndexSet, digitCount, totalCount)
    totalCount = totalCount + 1

print("Splitting numbers 0-9...")

# SIMILAR ARRAYS TO STORE CONDENSED IMAGES ASSOCIATED WITH EACH DIGIT
smallPic = np.zeros((4, 4))
digitImageSet = np.zeros((10, 28000, 4, 4))
digitImageIndexSet = np.zeros((10, 28000), dtype = int)

# KEEP TRACK OF HOW MANY OF EACH IMAGE (AND TOTAL) ARE PROCESSED
digitImageCount = np.zeros(10, dtype = int)
totalImageCount = 0

print(f"\nPreprocessing images...")

from alive_progress import alive_bar
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
            if totalImageCount in digitIndexSet[digit]:
                addDigit(digit, smallPic, digitImageSet, digitImageIndexSet, digitImageCount, totalImageCount)
                break

        totalImageCount = totalImageCount + 1
        bar()

# NUMBER OF REPEATS COVERS APPROX 5% OF IMAGES
T = 1400

# STORE T IMAGES CORRESPONDING TO EACH DIGIT
sampleImageSet = np.zeros((10, T, 4, 4))
sampleImageList = np.zeros((14000, 4, 4))
sizeUniqueImageSet = np.zeros(10)
overallCount = 0

print(f"\nFinding unique images...")

for D in range(0, 10):

    # RANDOMLY SAMPLE T INDICES FROM EACH DIGIT SET
    randomIndices = random.sample(range(0, 28000), T)
    sampleCount = 0
    
    for index in randomIndices:

        # EXTRACT EACH IMAGE CORRESPONDING TO EACH OF THE T INDICES AND SAVE IN NEW STRUCTURE
        randomImage = digitImageSet[D, index]
        sampleImageSet[D, sampleCount] = randomImage
        sampleImageList[overallCount] = randomImage
        sampleCount = sampleCount + 1
        overallCount = overallCount + 1
        
    # FIND COUNTS OF ALL UNIQUE IMAGES IN SAMPLE IMAGE SET
    uniqueImageSet = np.unique(sampleImageSet[D], axis = 0)
    sizeUniqueImageSet[D] = len(uniqueImageSet)

# FIND COUNTS OF UNIQUE IMAGES IN SAMPLE IMAGE LIST
uniqueImageList = np.unique(sampleImageList, axis = 0)
sizeUniqueImageList = len(uniqueImageList)

# DOMAIN FOR EACH DIGIT DISTRIBUTION IS NUMBER OF UNIQUE IMAGES
U = 207

# FIND AND STORE FREQUENCIES OF UNIQUE IMAGES FOR EACH DIGIT
uDigitImageSet = np.zeros((10, U, 4, 4))
uDigitFreqSet = np.zeros((10, U))
uDigitProbsSet = np.zeros((10, U))

print(f"Creating probability distributions...")

# SMOOTHING PARAMETER: 0.1 AND 1 ARE TOO LARGE
alpha = 0.01

for D in range(0, 10):
    uniqueCount = 0

    # STORE IMAGE AND SMOOTHED PROBABILITY AS WELL AS FREQUENCY
    for image in uniqueImageList:
        where = np.where(np.all(image == sampleImageSet[D], axis = (1, 2)))
        freq = len(where[0])
        uDigitImageSet[D, uniqueCount] = image
        uDigitFreqSet[D, uniqueCount] = int(freq)
        uDigitProbsSet[D, uniqueCount] = float((freq + alpha)/(T + (alpha*(sizeUniqueImageSet[D]))))
        uniqueCount = uniqueCount + 1

KLDiv = np.zeros((10, 10, U))
sumKLDiv = np.zeros((10, 10))
KList = list()
CDList = list()

print(f"Computing KL divergence...")

# FOR EACH COMPARISON DIGIT COMPUTE KLD FOR ALL DIGITS
for C in range(0, 10):
    for D in range(0, 10):
        for i in range(0, U):
            KLDiv[C, D, i] = uDigitProbsSet[D, i] * np.log((uDigitProbsSet[D, i]) / (uDigitProbsSet[C, i]))

        # ELIMINATE ALL ZERO VALUES WHEN DIGITS ARE IDENTICAL
        if sum(KLDiv[C, D]) != 0.0:
            sumKLDiv[C, D] = (sum(KLDiv[C, D]))
            KList.append(sum(KLDiv[C, D]))
            CDList.append((C, D))

KLDict = dict(zip(KList, CDList))
orderedKLDict = OrderedDict(sorted(KLDict.items()))
datafile = open("kl_divergence_in_order.txt", "w")
datafile.write("KL Divergence In Order\n")
datafile.write("Smaller corresponds to more similar digits\n\n")

for i in orderedKLDict:
    datafile.write(f"{i} : {orderedKLDict[i]}\n")

# SHOW ALL RANDOM IMAGES AT THE SAME TIME
fig, ax = plt.subplots(2, 5, sharex = True, sharey = True)

plotCount = 0
for row in ax:
    for col in row:
        randomNumber = random.randint(0, 1399)
        col.imshow(sampleImageSet[plotCount, randomNumber], cmap = 'gray')
        col.set_title(f'Digit: {plotCount}')
        plotCount = plotCount + 1

plt.ion()
plt.show()
plt.pause(0.001)
input("\nPress [enter] to continue.")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Total runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Total runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")