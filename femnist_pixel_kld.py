"""Modules provide various time-related functions, provide both a high- and low-level
interface to the HDF5 library, and work with arrays in Python."""
import time
import matplotlib.pyplot as plt
import h5py
import numpy as np
np.set_printoptions(suppress=True)

# initialising start time and seed for random sampling
startTime = time.perf_counter()
print("\nStarting...")
np.random.seed(106742)

# from HDF5-FEMNIST by Xiao-Chenguang
# https://github.com/Xiao-Chenguang/HDF5-FEMNIST
# enables easy access and fast loading to the FEMNIST dataset from LEAF with the help of HDF5

# fetch hdf5 file from current directory
PATH = './data/write_all.hdf5'
file = h5py.File(PATH, 'r')

# create list storing images and labels of each writer
writers = sorted(file.keys())
numWriters = len(writers)
numSampledWriters = int(numWriters / 20)

# randomly sample 5% of writers without replacement
sampledWriters = np.random.choice(numWriters, numSampledWriters)
totalDigits = np.zeros(10, dtype = int)

# add up how many times each digit is featured
print("Preprocessing images...")
for i in sampledWriters:
    tempDataset = file[writers[i]]

    for pic in range(len(tempDataset['labels'])):

        for d in range(10):
            if tempDataset['labels'][pic] == d:
                totalDigits[d] = totalDigits[d] + 1

# create image store of appropriate dimensions for each digit
zeroSet = np.zeros((totalDigits[0], 4, 4), dtype = int)
oneSet = np.zeros((totalDigits[1], 4, 4), dtype = int)
twoSet = np.zeros((totalDigits[2], 4, 4), dtype = int)
threeSet = np.zeros((totalDigits[3], 4, 4), dtype = int)
fourSet = np.zeros((totalDigits[4], 4, 4), dtype = int)
fiveSet = np.zeros((totalDigits[5], 4, 4), dtype = int)
sixSet = np.zeros((totalDigits[6], 4, 4), dtype = int)
sevenSet = np.zeros((totalDigits[7], 4, 4), dtype = int)
eightSet = np.zeros((totalDigits[8], 4, 4), dtype = int)
nineSet = np.zeros((totalDigits[9], 4, 4), dtype = int)

# to store condensed image and frequency of each digit
smallPic = np.zeros((4, 4), dtype = int)
digitCount = np.zeros(10, dtype = int)

def add_digit(dset):
    """Method to add digit to set corresponding to label."""
    dset[digitCount[label]] = smallPic

for i in sampledWriters:

    tempDataset = file[writers[i]]
    PIC_COUNT = 0

    for pic in tempDataset['images']:

        # partition each image into 16 7x7 subimages
        for a in range(4):
            for b in range(4):
                subImage = pic[7*a : 7*(a + 1), 7*b : 7*(b + 1)]

                # save rounded mean of each subimage into corresponding cell of smallpic
                meanSubImage = np.mean(subImage)
                if meanSubImage >= 220:
                    smallPic[a, b] = 1
                else:
                    smallPic[a, b] = 0

        label = tempDataset['labels'][PIC_COUNT]

        # split images according to label
        if label == 0:
            add_digit(zeroSet)
        elif label == 1:
            add_digit(oneSet)
        elif label == 2:
            add_digit(twoSet)
        elif label == 3:
            add_digit(threeSet)
        elif label == 4:
            add_digit(fourSet)
        elif label == 5:
            add_digit(fiveSet)
        elif label == 6:
            add_digit(sixSet)
        elif label == 7:
            add_digit(sevenSet)
        elif label == 8:
            add_digit(eightSet)
        elif label == 9:
            add_digit(nineSet)

        digitCount[label] = digitCount[label] + 1
        PIC_COUNT = PIC_COUNT + 1

# plot a random example of each digit
fig, axs = plt.subplots(3, 4, figsize = (5, 3))

for i in range(3):
    for j in range(4):
        axs[i, j].axis('off')

# check whether images and labels match
zeroIndex = np.random.randint(len(zeroSet))
print(f"\nExample of zeroSet:\n{zeroSet[zeroIndex]}")
axs[0, 0].imshow(zeroSet[zeroIndex])

oneIndex = np.random.randint(len(oneSet))
print(f"\nExample of oneSet:\n{oneSet[oneIndex]}")
axs[0, 1].imshow(oneSet[oneIndex])

twoIndex = np.random.randint(len(twoSet))
print(f"\nExample of twoSet:\n{twoSet[twoIndex]}")
axs[0, 2].imshow(twoSet[twoIndex])

threeIndex = np.random.randint(len(threeSet))
print(f"\nExample of threeSet:\n{threeSet[threeIndex]}")
axs[0, 3].imshow(threeSet[threeIndex])

fourIndex = np.random.randint(len(fourSet))
print(f"\nExample of fourSet:\n{fourSet[fourIndex]}")
axs[1, 0].imshow(fourSet[fourIndex])

fiveIndex = np.random.randint(len(fiveSet))
print(f"\nExample of fiveSet:\n{fiveSet[fiveIndex]}")
axs[1, 1].imshow(fiveSet[fiveIndex])

sixIndex = np.random.randint(len(sixSet))
print(f"\nExample of sixSet:\n{sixSet[sixIndex]}")
axs[1, 2].imshow(sixSet[sixIndex])

sevenIndex = np.random.randint(len(sevenSet))
print(f"\nExample of sevenSet:\n{sevenSet[sevenIndex]}")
axs[1, 3].imshow(sevenSet[sevenIndex])

eightIndex = np.random.randint(len(eightSet))
print(f"\nExample of eightSet:\n{eightSet[eightIndex]}")
axs[2, 1].imshow(eightSet[eightIndex])

nineIndex = np.random.randint(len(nineSet))
print(f"\nExample of nineSet:\n{nineSet[nineIndex]}\n")
axs[2, 2].imshow(nineSet[nineIndex])

plt.show()
