import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(3957204)

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

# SIMILAR ARRAYS TO STORE IMAGES ASSOCIATED WITH EACH DIGIT
digitImageSet = np.zeros((10, 28000, 28, 28), dtype = int)
digitImageIndexSet = np.zeros((10, 28000), dtype = int)

# KEEP TRACK OF HOW MANY OF EACH IMAGE (AND TOTAL) ARE PROCESSED
digitImageCount = np.zeros(10, dtype = int)
totalImageCount = 0

for pic in images:

    # SPLIT IMAGES BY ASSOCIATION WITH PARTICULAR LABEL
    for digit in range(0, 10):
        if totalImageCount in digitIndexSet[digit]:
            addDigit(digit, pic, digitImageSet, digitImageIndexSet, digitImageCount, totalImageCount)
            break

    totalImageCount = totalImageCount + 1

# NUMBER OF REPEATS
T = 100

# PREPARE TEMPLATES FOR AVERAGE IMAGES AND TOTAL DIFFERENCES BETWEEN IMAGES
firstDiff = np.zeros((T, 28, 28))
secondDiff = np.zeros((T, 28, 28))
firstTotalDiff = np.zeros(T)
secondTotalDiff = np.zeros(T)
firstAvgDiff = np.zeros(T)
secondAvgDiff = np.zeros(T)
firstImage = np.zeros((28, 28))
secondImage = np.zeros((28, 28))

for A in range(T):

    # CHOOSE RANDOM IMAGE FROM EACH OF LABELS ONE TWO AND THREE
    randomOne = random.randint(0, 27999)
    randomTwo = random.randint(0, 27999)
    randomThree = random.randint(0, 27999)

    firstOne = digitImageSet[1, randomOne]
    firstTwo = digitImageSet[2, randomTwo]
    firstThree = digitImageSet[3, randomThree]

    # LOOP ACROSS ALL PIXELS IN EACH IMAGE
    for i in range(28):
        for j in range(28):

            # COMPUTE MIDPOINT BETWEEN PIXELS AND ADD IT TO TOTAL DIFFERENCE
            firstDiff[A, i, j] = abs(0.5*(firstOne[i, j] - firstTwo[i, j]))
            secondDiff[A, i, j] = abs(0.5*(firstOne[i, j] - firstThree[i, j]))
            firstTotalDiff[A] = firstTotalDiff[A] + firstDiff[A, i, j]
            secondTotalDiff[A] = secondTotalDiff[A] + secondDiff[A, i, j]
            firstImage[i, j] = firstImage[i, j] + firstDiff[A, i, j]
            secondImage[i, j] = secondImage[i, j] + secondDiff[A, i, j]

    # DIVIDE BY NUMBER OF PIXELS IN AN IMAGE
    firstAvgDiff[A] = firstTotalDiff[A] / 784
    secondAvgDiff[A] = secondTotalDiff[A] / 784

print(firstImage[13])
print(secondImage[13])

for i in range(28):
    for j in range(28):
        firstImage[i, j] = firstImage[i, j] / T
        secondImage[i, j] = secondImage[i, j] / T

print(sum(firstTotalDiff)/T)
print(sum(secondTotalDiff)/T)
print(round(sum(firstAvgDiff)/T, 4))
print(round(sum(secondAvgDiff)/T, 4))

print(firstImage[13])
print(secondImage[13])

# SHOW BOTH AVERAGE IMAGES AT THE SAME TIME
fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (12, 6))
ax1.imshow(firstImage, cmap = 'gray')
ax2.imshow(secondImage, cmap = 'gray')
plt.show()