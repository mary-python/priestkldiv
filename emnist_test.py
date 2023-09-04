import numpy as np
import random
import torch
torch.manual_seed(85)
import torch.distributions as dis
from math import log
import tensorflow as tf
tf.random.set_seed(475)
import tensorflow_probability as tfp
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

# NUMBER OF REPEATS COVERS APPROX 5% OF DEVICES
T = 180

# PREPARE TEMPLATES FOR AVERAGE IMAGES AND TOTAL DIFFERENCES BETWEEN IMAGES
oneDiff = np.zeros((T, 10, 4, 4))
totalDiff = np.zeros((T, 10))
avgDiff = np.zeros((T, 10))
imageDiff = np.zeros((10, 4, 4))

for A in range(T):

    # COMPARE DIGIT "1" WITH ALL OTHER DIGITS INCLUDING ITSELF
    for D in range(10):

        # CHOOSE RANDOM IMAGES FROM FIRST HALF OF "1" SET AND SECOND HALF OF "D" SET
        randomOne = random.randint(0, 13999)
        randomOther = random.randint(14000, 27999)
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

sumDiff = np.zeros(10)

for D in range(10):
    for i in range(4):
        for j in range(4):
            imageDiff[D, i, j] = imageDiff[D, i, j] / T
    
    # SUM OVER ALL REPEATS FOR EACH DIGIT
    sumDiff[D] = np.sum(avgDiff, axis = 0)
    print(f'{avgDiff[D]}')
    print(sumDiff[D])

logr = np.zeros(10)
k3 = np.zeros(10)
KLDest1 = np.zeros((10, 12))
KLDest2 = np.zeros((10, 12))

# NORMAL DISTRIBUTION AROUND DIFFERENCE OF "1" SET
p = dis.Normal(loc = sumDiff[1], scale = 1)

for D in range(10):

    # NORMAL DISTRIBUTION AROUND DIFFERENCE OF COMPARISON SET
    q = dis.Normal(loc = sumDiff[D], scale = 1)
    qT = torch.numel(sumDiff[D])
    truekl = dis.kl_divergence(p, q)
    print('true', truekl)

    # COMPUTE LOGR AND K3
    logr[D] = (p.log_prob(sumDiff[D]) - q.log_prob(sumDiff[D]))
    k3[D] = ((logr[D].exp() - 1) - logr[D])

    print(f'{logr[D]}')
    print(k3[D])

    # ADD LAPLACE AND GAUSSIAN NOISE
    epsset = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]
    dta = 0.1
    a = 0
    b1 = log(2)
    b2 = 2*((log(1.25))/dta)*b1

    noise1 = tfp.distributions.Laplace(loc=a, scale=b1)
    noise2 = tfp.distributions.Normal(loc=a, scale=b2)
    
    for eps in epsset:

        k3noise1 = list()
        k3noise2 = list()

        # FIND AVERAGE OF 10 NOISE TERMS FOR EACH
        for j in range(0, 10):
            k3noise1.append(k3[D] + (noise1.sample(sample_shape=qT, seed=12))/eps)
            k3noise2.append(k3[D] + (noise2.sample(sample_shape=qT, seed=12))/eps)

        average1 = sum(k3noise1) / len(k3noise1)
        average2 = sum(k3noise2) / len(k3noise2)
        KLDest1[D, eps] = abs(np.mean(average1) - truekl)
        KLDest2[D, eps] = abs(np.mean(average2) - truekl)

# PLOT EPSILON VS KLD ERROR FOR EACH COMPARISON
fig, ax = plt.subplots(2, 5, sharex = True, sharey = True)

plotCount = 0
for row in ax:
    for col in row:
        col.plot(epsset, KLDest1[plotCount], label = f'Laplace dist')
        col.plot(epsset, KLDest2[plotCount], label = f'Gaussian dist')
        col.title(f'Comparing 1 and {plotCount}')
        col.legend(loc = 'best')
        plotCount = plotCount + 1

plt.show()

# SHOW ALL AVERAGE IMAGES AT THE SAME TIME
fig, ax = plt.subplots(2, 5, sharex = True, sharey = True)

plotCount = 0
for row in ax:
    for col in row:
        col.imshow(imageDiff[plotCount], cmap = 'gray')
        col.title(f'Average image of {plotCount}')
        plotCount = plotCount + 1

plt.show()