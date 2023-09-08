import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(3957204)

# LOAD TRAINING AND TEST SAMPLES FOR 'DIGITS' SUBSET
from emnist import extract_training_samples, extract_test_samples
images1, labels1 = extract_training_samples('digits')
images2, labels2 = extract_test_samples('digits')

# COMBINE TRAINING AND TEST SAMPLES INTO ONE NP ARRAY
images = np.concatenate((images1, images2))
labels = np.concatenate((labels1, labels2))

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

# SIMILAR ARRAYS TO STORE CONDENSED IMAGES ASSOCIATED WITH EACH DIGIT
smallPic = np.zeros((4, 4))
digitImageSet = np.zeros((10, 28000, 4, 4))
digitImageIndexSet = np.zeros((10, 28000), dtype = int)

# KEEP TRACK OF HOW MANY OF EACH IMAGE (AND TOTAL) ARE PROCESSED
digitImageCount = np.zeros(10, dtype = int)
totalImageCount = 0

for pic in images:

    # PARTITION EACH IMAGE INTO 16 7x7 SUBIMAGES
    for i in range(4):
        for j in range(4):
            subImage = pic[7*i : 7*(i + 1), 7*j : 7*(j + 1)]

            # SAVE ROUNDED MEAN OF EACH SUBIMAGE INTO CORRESPONDING CELL OF SMALLPIC
            meanSubImage = np.mean(subImage)
            smallPic[i, j] = 2*round(meanSubImage / 2)

    # SPLIT IMAGES BY ASSOCIATION WITH PARTICULAR LABEL
    for digit in range(0, 10):
        if totalImageCount in digitIndexSet[digit]:
            addDigit(digit, smallPic, digitImageSet, digitImageIndexSet, digitImageCount, totalImageCount)
            break

    totalImageCount = totalImageCount + 1

# NUMBER OF REPEATS COVERS APPROX 5% OF IMAGES
T = 1400

# STORE T IMAGES CORRESPONDING TO EACH DIGIT
sampleImageSet = np.zeros((10, T, 4, 4))
histogramSet = np.zeros((10, T, 4, 4))

for D in range(0, 10):

    # RANDOMLY SAMPLE T INDICES FROM EACH DIGIT SET
    randomIndices = random.sample(range(0, 28000), T)
    sampleImageCount = 0

    for index in randomIndices:

        # EXTRACT EACH IMAGE CORRESPONDING TO EACH OF THE T INDICES AND SAVE IN NEW STRUCTURE
        randomImage = digitImageSet[D, index]
        sampleImageSet[D, sampleImageCount] = randomImage
        sampleImageCount = sampleImageCount + 1

    # COMPUTE MULTIDIMENSIONAL HISTOGRAM OF SAMPLED IMAGES
    tempHist, bins = np.histogramdd(sampleImageSet[D])
    histogramSet[D] = tempHist

    for i in range(0, 3):
        print(f'Bins along dimension {i}: {bins[i]}')

    print(f'Counts after binning: {histogramSet[D]}')

# SHOW ALL RANDOM IMAGES AT THE SAME TIME
fig, ax = plt.subplots(2, 5, sharex = True, sharey = True)

plotCount = 0
for row in ax:
    for col in row:
        randomNumber = random.randint(0, 1399)
        col.imshow(sampleImageSet[plotCount, randomNumber], cmap = 'gray')
        col.set_title(f'Digit: {plotCount}')
        plotCount = plotCount + 1

plt.show()