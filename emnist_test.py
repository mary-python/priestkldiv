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

# NUMBER OF REPEATS COVERS APPROX 5% OF IMAGES
T = 1400

# STORE T IMAGES CORRESPONDING TO EACH DIGIT
sampleImageSet = np.zeros((10, T, 4, 4))
sizeUniqueImages = np.zeros(10)
imageList = list()
freqList = list()

for D in range(0, 10):

    # RANDOMLY SAMPLE T INDICES FROM EACH DIGIT SET
    randomIndices = random.sample(range(0, 28000), T)
    sampleImageCount = 0

    for index in randomIndices:

        # EXTRACT EACH IMAGE CORRESPONDING TO EACH OF THE T INDICES AND SAVE IN NEW STRUCTURE
        randomImage = digitImageSet[D, index]
        sampleImageSet[D, sampleImageCount] = randomImage
        sampleImageCount = sampleImageCount + 1
    
    # FIND COUNTS OF ALL UNIQUE IMAGES IN SAMPLE IMAGE SET
    uniqueImages = np.unique(sampleImageSet[D], axis = 0)
    sizeUniqueImages[D] = len(uniqueImages)
    print(len(uniqueImages))

    for image in uniqueImages:
        where = np.where(np.all(image == sampleImageSet[D], axis = (1, 2)))
        frequency = len(where[0])
        imageList.append(image)
        freqList.append(frequency)

print(len(imageList))
print(len(freqList))

cumFreqList = np.zeros(10)

for D in range(0, 10):
    if D == 0:
        cumFreqList[D] = sizeUniqueImages[D]
    else:
        cumFreqList[D] = sizeUniqueImages[D] + cumFreqList[D - 1]

    print(cumFreqList)

zeroImageList = imageList[0:(cumFreqList[0]-1)]
oneImageList = imageList[cumFreqList[0]:(cumFreqList[1]-1)]
twoImageList = imageList[cumFreqList[1]:(cumFreqList[2]-1)]
threeImageList = imageList[cumFreqList[2]:(cumFreqList[3]-1)]
fourImageList = imageList[cumFreqList[3]:(cumFreqList[4]-1)]
fiveImageList = imageList[cumFreqList[4]:(cumFreqList[5]-1)]
sixImageList = imageList[cumFreqList[5]:(cumFreqList[6]-1)]
sevenImageList = imageList[cumFreqList[6]:(cumFreqList[7]-1)]
eightImageList = imageList[cumFreqList[7]:(cumFreqList[8]-1)]
nineImageList = imageList[cumFreqList[8]:(cumFreqList[9]-1)]

print(len(zeroImageList))
print(len(oneImageList))
print(len(twoImageList))
print(len(threeImageList))
print(len(fourImageList))
print(len(fiveImageList))
print(len(sixImageList))
print(len(sevenImageList))
print(len(eightImageList))
print(len(nineImageList))

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