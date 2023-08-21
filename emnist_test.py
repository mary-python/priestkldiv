# FOR NUMPY ARRAYS AND PLOTS
import numpy as np
import matplotlib.pyplot as plt

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
def addDigit(dg, set, iset, count, tc):
    set[dg, count[dg]] = digit
    iset[dg, count[dg]] = tc
    count[dg] = count[dg] + 1

# SPLIT NUMBERS 0-9
for digit in labels:
    
    # CALL FUNCTION DEFINED ABOVE
    addDigit(digit, digitSet, digitIndexSet, digitCount, totalCount)
    totalCount = totalCount + 1

print(digitSet.shape)
print(digitIndexSet.shape)
print(digitCount)
print(totalCount)

# SIMILAR ARRAYS TO STORE IMAGES ASSOCIATED WITH EACH DIGIT
digitImageSet = np.zeros((10, 28000, 28, 28), dtype = int)
digitImageIndexSet = np.zeros((10, 28000), dtype = int)

# KEEP TRACK OF HOW MANY OF EACH IMAGE (AND TOTAL) ARE PROCESSED
digitImageCount = np.zeros(10, dtype = int)
totalImageCount = 0

for pic in images:

    for digit in range(0, 10):
        if totalImageCount in digitIndexSet[digit]:
            addDigit(digit, digitImageSet, digitImageIndexSet, digitImageCount, totalImageCount)
            break

    totalImageCount = totalImageCount + 1

print(digitImageSet.shape)
print(digitImageIndexSet.shape)
print(digitImageCount)
print(totalImageCount)

print(digitImageSet[0, 0])
print(digitImageSet[0, 0, 0])
print(digitImageSet[0, 0, 0, 0])

plt.gray()
plt.imshow(digitImageSet[0, 0])