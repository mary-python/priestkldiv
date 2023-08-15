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
evenSet = np.zeros(140000)
oddSet = np.zeros(140000)
evenIndexSet = np.zeros(140000)
oddIndexSet = np.zeros(140000)

# KEEP TRACK OF HOW MANY ODD/EVEN/TOTAL DIGITS ARE PROCESSED
evenCount = 0
oddCount = 0
count = 0

# SPLIT NUMBERS 0-9 INTO ODD AND EVEN
for digit in labels:
    
    # IF NO REMAINDER WHEN DIVIDED BY 2 THEN EVEN
    if (digit % 2) == 0:
        evenSet[evenCount] = digit
        evenIndexSet[evenCount] = count
        evenCount = evenCount + 1

    # IF REMAINDER WHEN DIVIDED BY 2 THEN ODD
    else:
        oddSet[oddCount] = digit
        oddIndexSet[oddCount] = count
        oddCount = oddCount + 1

    count = count + 1

# NUMPY ARRAYS WITH DIMENSION OF IMAGES
evenImageSet = np.zeros((140000, 28, 28))
oddImageSet = np.zeros((140000, 28, 28))

# KEEP TRACK OF HOW MANY ODD/EVEN/TOTAL IMAGES ARE PROCESSED
evenImageCount = 0
oddImageCount = 0
imageCount = 0

for pic in images:

    # IF ASSOCIATED WITH AN EVEN LABEL THEN ADD TO EVEN IMAGE SET
    if imageCount in evenIndexSet:
        evenImageSet[evenImageCount] = pic
        evenImageCount = evenImageCount + 1

    # IF ASSOCIATED WITH AN ODD LABEL THEN ADD TO ODD IMAGE SET
    else:
        oddImageSet[oddImageCount] = pic
        oddImageCount = oddImageCount + 1

    imageCount = imageCount + 1

# CHECK WHETHER IMAGES ARE SPLIT EVENLY OR NOT
print(evenImageSet.shape)
print(oddImageSet.shape)