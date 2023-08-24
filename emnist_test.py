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
            smallPic[i, j] = 16*round(meanSubImage / 16)

    # SPLIT IMAGES BY ASSOCIATION WITH PARTICULAR LABEL
    for digit in range(0, 10):
        if totalImageCount in digitIndexSet[digit]:
            addDigit(digit, smallPic, digitImageSet, digitImageIndexSet, digitImageCount, totalImageCount)
            break

    totalImageCount = totalImageCount + 1

# NUMBER OF REPEATS
T = 100

# PREPARE TEMPLATES FOR AVERAGE IMAGES AND TOTAL DIFFERENCES BETWEEN IMAGES
oneDiff = np.zeros((T, 10, 4, 4))
totalDiff = np.zeros((T, 10))
avgDiff = np.zeros((T, 10))
imageDiff = np.zeros((10, 4, 4))

for A in range(T):

    # COMPARE DIGIT "1" WITH ALL OTHER DIGITS INCLUDING ITSELF
    for D in range(10):

        # CHOOSE RANDOM IMAGES FROM "1" SET AND "D" SET
        randomOne = random.randint(0, 27999)
        randomOther = random.randint(0, 27999)
        oneImage = digitImageSet[1, randomOne]
        otherImage = digitImageSet[D, randomOther]

        # LOOP ACROSS ALL SUBIMAGES IN EACH IMAGE
        for i in range(4):
            for j in range(4):

                # COMPUTE MIDPOINT BETWEEN SUBIMAGES AND ADD IT TO TOTAL DIFFERENCE
                oneDiff[A, D, i, j] = abs(0.5*(oneImage[i, j] - otherImage[i, j]))
                totalDiff[A, D] = totalDiff[A, D] + oneDiff[A, D, i, j]
                imageDiff[D, i, j] = imageDiff[D, i, j] + oneDiff[A, D, i, j]

        # DIVIDE BY NUMBER OF SUBIMAGES IN AN IMAGE
        avgDiff[A, D] = totalDiff[A, D] / 16

for D in range(10):
    for i in range(4):
        for j in range(4):
            imageDiff[D, i, j] = imageDiff[D, i, j] / T
    
    print(round(sum(avgDiff[D])/T, 4))
    print(imageDiff[D, 2])

# SHOW BOTH AVERAGE IMAGES AT THE SAME TIME
fig, ax = plt.subplots(2, 5, sharex = True, sharey = True)

plotCount = 0
for row in ax:
    for col in row:
        col.imshow(imageDiff[plotCount], cmap = 'gray')
        plotCount = plotCount + 1

plt.show()