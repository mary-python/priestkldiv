# FOR NUMPY ARRAYS
import numpy as np

# LOAD TRAINING AND TEST SAMPLES FOR 'DIGITS' SUBSET
from emnist import extract_training_samples, extract_test_samples
images1, labels1 = extract_training_samples('digits')
images2, labels2 = extract_test_samples('digits')

# COMBINE TRAINING AND TEST SAMPLES INTO ONE NP ARRAY
images = np.concatenate((images1, images2))
labels = np.concatenate((labels1, labels2))

# NUMPY ARRAYS WITH DIMENSION OF LABELS
zeroSet = np.zeros(28000)
oneSet = np.zeros(28000)
twoSet = np.zeros(28000)
threeSet = np.zeros(28000)
fourSet = np.zeros(28000)
fiveSet = np.zeros(28000)
sixSet = np.zeros(28000)
sevenSet = np.zeros(28000)
eightSet = np.zeros(28000)
nineSet = np.zeros(28000)

# TO STORE WHICH LABELS ARE ASSOCIATED WITH WHICH DIGIT
zeroIndexSet = np.zeros(28000)
oneIndexSet = np.zeros(28000)
twoIndexSet = np.zeros(28000)
threeIndexSet = np.zeros(28000)
fourIndexSet = np.zeros(28000)
fiveIndexSet = np.zeros(28000)
sixIndexSet = np.zeros(28000)
sevenIndexSet = np.zeros(28000)
eightIndexSet = np.zeros(28000)
nineIndexSet = np.zeros(28000)

# KEEP TRACK OF HOW MANY OF EACH DIGIT (AND TOTAL) ARE PROCESSED
zeroCount = 0
oneCount = 0
twoCount = 0
threeCount = 0
fourCount = 0
fiveCount = 0
sixCount = 0
sevenCount = 0
eightCount = 0
nineCount = 0
totalCount = 0

# SPLIT NUMBERS 0-9
for digit in labels:
    
    # ADD DIGIT TO SET, INDEX TO INDEX SET AND INCREMENT COUNT
    if digit == 0:
        zeroSet[zeroCount] = digit
        zeroIndexSet[zeroCount] = totalCount
        zeroCount = zeroCount + 1

    elif digit == 1:
        oneSet[oneCount] = digit
        oneIndexSet[oneCount] = totalCount
        oneCount = oneCount + 1

    elif digit == 2:
        twoSet[twoCount] = digit
        twoIndexSet[twoCount] = totalCount
        twoCount = twoCount + 1

    elif digit == 3:
        threeSet[threeCount] = digit
        threeIndexSet[threeCount] = totalCount
        threeCount = threeCount + 1

    elif digit == 4:
        fourSet[fourCount] = digit
        fourIndexSet[fourCount] = totalCount
        fourCount = fourCount + 1

    elif digit == 5:
        fiveSet[fiveCount] = digit
        fiveIndexSet[fiveCount] = totalCount
        fiveCount = fiveCount + 1

    elif digit == 6:
        sixSet[sixCount] = digit
        sixIndexSet[sixCount] = totalCount
        sixCount = sixCount + 1

    elif digit == 7:
        sevenSet[sevenCount] = digit
        sevenIndexSet[sevenCount] = totalCount
        sevenCount = sevenCount + 1

    elif digit == 8:
        eightSet[eightCount] = digit
        eightIndexSet[eightCount] = totalCount
        eightCount = eightCount + 1

    elif digit == 9:
        nineSet[nineCount] = digit
        nineIndexSet[nineCount] = totalCount
        nineCount = nineCount + 1

    totalCount = totalCount + 1

print(zeroSet.shape)
print(oneSet.shape)
print(twoSet.shape)
print(threeSet.shape)
print(fourSet.shape)
print(fiveSet.shape)
print(sixSet.shape)
print(sevenSet.shape)
print(eightSet.shape)
print(nineSet.shape)

print(zeroIndexSet.shape)
print(oneIndexSet.shape)
print(twoIndexSet.shape)
print(threeIndexSet.shape)
print(fourIndexSet.shape)
print(fiveIndexSet.shape)
print(sixIndexSet.shape)
print(sevenIndexSet.shape)
print(eightIndexSet.shape)
print(nineIndexSet.shape)

print(zeroCount)
print(oneCount)
print(twoCount)
print(threeCount)
print(fourCount)
print(fiveCount)
print(sixCount)
print(sevenCount)
print(eightCount)
print(nineCount)
print(totalCount)

# NUMPY ARRAYS WITH DIMENSION OF IMAGES
zeroImageSet = np.zeros((28000, 28, 28))
oneImageSet = np.zeros((28000, 28, 28))
twoImageSet = np.zeros((28000, 28, 28))
threeImageSet = np.zeros((28000, 28, 28))
fourImageSet = np.zeros((28000, 28, 28))
fiveImageSet = np.zeros((28000, 28, 28))
sixImageSet = np.zeros((28000, 28, 28))
sevenImageSet = np.zeros((28000, 28, 28))
eightImageSet = np.zeros((28000, 28, 28))
nineImageSet = np.zeros((28000, 28, 28))

# TO STORE WHICH IMAGES ARE ASSOCIATED WITH WHICH DIGIT
zeroImageIndexSet = np.zeros(28000)
oneImageIndexSet = np.zeros(28000)
twoImageIndexSet = np.zeros(28000)
threeImageIndexSet = np.zeros(28000)
fourImageIndexSet = np.zeros(28000)
fiveImageIndexSet = np.zeros(28000)
sixImageIndexSet = np.zeros(28000)
sevenImageIndexSet = np.zeros(28000)
eightImageIndexSet = np.zeros(28000)
nineImageIndexSet = np.zeros(28000)

# KEEP TRACK OF HOW MANY OF EACH IMAGE (AND TOTAL) ARE PROCESSED
zeroImageCount = 0
oneImageCount = 0
twoImageCount = 0
threeImageCount = 0
fourImageCount = 0
fiveImageCount = 0
sixImageCount = 0
sevenImageCount = 0
eightImageCount = 0
nineImageCount = 0
totalImageCount = 0

for pic in images:

    # ADD IMAGE TO SET, INDEX TO INDEX SET AND INCREMENT COUNT
    if totalImageCount in zeroIndexSet:
        zeroImageSet[zeroImageCount] = pic
        zeroImageIndexSet[zeroImageCount] = totalImageCount
        zeroImageCount = zeroImageCount + 1

    elif totalImageCount in oneIndexSet:
        oneImageSet[oneImageCount] = pic
        oneImageIndexSet[oneImageCount] = totalImageCount
        oneImageCount = oneImageCount + 1
    
    elif totalImageCount in twoIndexSet:
        twoImageSet[twoImageCount] = pic
        twoImageIndexSet[twoImageCount] = totalImageCount
        twoImageCount = twoImageCount + 1
    
    elif totalImageCount in threeIndexSet:
        threeImageSet[threeImageCount] = pic
        threeImageIndexSet[threeImageCount] = totalImageCount
        threeImageCount = threeImageCount + 1
    
    elif totalImageCount in fourIndexSet:
        fourImageSet[fourImageCount] = pic
        fourImageIndexSet[fourImageCount] = totalImageCount
        fourImageCount = fourImageCount + 1
    
    elif totalImageCount in fiveIndexSet:
        fiveImageSet[fiveImageCount] = pic
        fiveImageIndexSet[fiveImageCount] = totalImageCount
        fiveImageCount = fiveImageCount + 1
    
    elif totalImageCount in sixIndexSet:
        sixImageSet[sixImageCount] = pic
        sixImageIndexSet[sixImageCount] = totalImageCount
        sixImageCount = sixImageCount + 1
    
    elif totalImageCount in sevenIndexSet:
        sevenImageSet[sevenImageCount] = pic
        sevenImageIndexSet[sevenImageCount] = totalImageCount
        sevenImageCount = sevenImageCount + 1
    
    elif totalImageCount in eightIndexSet:
        eightImageSet[eightImageCount] = pic
        eightImageIndexSet[eightImageCount] = totalImageCount
        eightImageCount = eightImageCount + 1
    
    elif totalImageCount in nineIndexSet:
        nineImageSet[nineImageCount] = pic
        nineImageIndexSet[nineImageCount] = totalImageCount
        nineImageCount = nineImageCount + 1

    totalImageCount = totalImageCount + 1

print(zeroImageSet.shape)
print(oneImageSet.shape)
print(twoImageSet.shape)
print(threeImageSet.shape)
print(fourImageSet.shape)
print(fiveImageSet.shape)
print(sixImageSet.shape)
print(sevenImageSet.shape)
print(eightImageSet.shape)
print(nineImageSet.shape)

print(zeroImageIndexSet.shape)
print(oneImageIndexSet.shape)
print(twoImageIndexSet.shape)
print(threeImageIndexSet.shape)
print(fourImageIndexSet.shape)
print(fiveImageIndexSet.shape)
print(sixImageIndexSet.shape)
print(sevenImageIndexSet.shape)
print(eightImageIndexSet.shape)
print(nineImageIndexSet.shape)

print(zeroImageCount)
print(oneImageCount)
print(twoImageCount)
print(threeImageCount)
print(fourImageCount)
print(fiveImageCount)
print(sixImageCount)
print(sevenImageCount)
print(eightImageCount)
print(nineImageCount)
print(totalImageCount)