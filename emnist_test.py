import time
import numpy as np
import random
import matplotlib.pyplot as plt

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

    for image in uniqueImages:
        where = np.where(np.all(image == sampleImageSet[D], axis = (1, 2)))
        frequency = len(where[0])
        imageList.append(image)
        freqList.append(frequency)

print(f"\nNumber of unique images for each digit: {sizeUniqueImages}")

cumFreqList = np.zeros(10)

for D in range(0, 10):
    if D == 0:
        cumFreqList[D] = sizeUniqueImages[D]
    else:
        cumFreqList[D] = sizeUniqueImages[D] + cumFreqList[D - 1]

zeroImageList = imageList[0:int(cumFreqList[0])]
oneImageList = imageList[int(cumFreqList[0]):int(cumFreqList[1])]
twoImageList = imageList[int(cumFreqList[1]):int(cumFreqList[2])]
threeImageList = imageList[int(cumFreqList[2]):int(cumFreqList[3])]
fourImageList = imageList[int(cumFreqList[3]):int(cumFreqList[4])]
fiveImageList = imageList[int(cumFreqList[4]):int(cumFreqList[5])]
sixImageList = imageList[int(cumFreqList[5]):int(cumFreqList[6])]
sevenImageList = imageList[int(cumFreqList[6]):int(cumFreqList[7])]
eightImageList = imageList[int(cumFreqList[7]):int(cumFreqList[8])]
nineImageList = imageList[int(cumFreqList[8]):int(cumFreqList[9])]

zeroFreqList = freqList[0:int(cumFreqList[0])]
oneFreqList = freqList[int(cumFreqList[0]):int(cumFreqList[1])]
twoFreqList = freqList[int(cumFreqList[1]):int(cumFreqList[2])]
threeFreqList = freqList[int(cumFreqList[2]):int(cumFreqList[3])]
fourFreqList = freqList[int(cumFreqList[3]):int(cumFreqList[4])]
fiveFreqList = freqList[int(cumFreqList[4]):int(cumFreqList[5])]
sixFreqList = freqList[int(cumFreqList[5]):int(cumFreqList[6])]
sevenFreqList = freqList[int(cumFreqList[6]):int(cumFreqList[7])]
eightFreqList = freqList[int(cumFreqList[7]):int(cumFreqList[8])]
nineFreqList = freqList[int(cumFreqList[8]):int(cumFreqList[9])]

zeroDistProbsList = [freq/T for freq in zeroFreqList]
oneDistProbsList = [freq/T for freq in oneFreqList]
twoDistProbsList = [freq/T for freq in twoFreqList]
threeDistProbsList = [freq/T for freq in threeFreqList]
fourDistProbsList = [freq/T for freq in fourFreqList]
fiveDistProbsList = [freq/T for freq in fiveFreqList]
sixDistProbsList = [freq/T for freq in sixFreqList]
sevenDistProbsList = [freq/T for freq in sevenFreqList]
eightDistProbsList = [freq/T for freq in eightFreqList]
nineDistProbsList = [freq/T for freq in nineFreqList]

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
input("Press [enter] to continue.")

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Total runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Total runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")