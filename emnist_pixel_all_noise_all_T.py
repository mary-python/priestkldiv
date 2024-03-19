"""Modules provide various time-related functions, generate pseudo-random numbers,
compute the natural logarithm of a number, remember the order in which items are added,
have cool visual feedback of the current throughput, create static, animated,
and interactive visualisations, provide functionality to automatically download
and cache the EMNIST dataset, compute the mean of a list quickly and accurately,
work with arrays, and carry out fast numerical computations in Python."""
import time
import random
from math import log
from collections import OrderedDict
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
from statistics import fmean
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
np.set_printoptions(suppress = True)
np.seterr(divide = 'ignore', invalid = 'ignore')
tf.random.set_seed(638)

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter()
print("\nStarting...")
random.seed(3249583)
np.random.seed(107642)

# LOAD TRAINING AND TEST SAMPLES FOR 'DIGITS' SUBSET
images1, labels1 = extract_training_samples('digits')
images2, labels2 = extract_test_samples('digits')

# COMBINE TRAINING AND TEST SAMPLES INTO ONE NP ARRAY
images = np.concatenate((images1, images2))
labels = np.concatenate((labels1, labels2))
print("Loading training and test samples...")

# NUMPY ARRAYS TO STORE LABELS ASSOCIATED WITH WHICH DIGIT
digitSet = np.zeros((10, 28000), dtype = int)
digitIndexSet = np.zeros((10, 28000), dtype = int)

# REPORT FREQUENCY OF EACH DIGIT (AND TOTAL)
digitFreq = np.zeros(10, dtype = int)
TOTAL_FREQ = 0

def add_digit(dg, im, imset, ixset, freq, tc):
    """Method adds digit to set, index to index set and increments freq."""
    imset[dg, freq[dg]] = im
    ixset[dg, freq[dg]] = tc
    freq[dg] = freq[dg] + 1

# SPLIT NUMBERS 0-9
for digit in labels:

    # CALL FUNCTION DEFINED ABOVE
    add_digit(digit, digit, digitSet, digitIndexSet, digitFreq, TOTAL_FREQ)
    TOTAL_FREQ = TOTAL_FREQ + 1

print("Splitting numbers 0-9...")

# SIMILAR ARRAYS TO STORE CONDENSED IMAGES ASSOCIATED WITH EACH DIGIT
smallPic = np.zeros((4, 4))
digitImSet = np.zeros((10, 28000, 4, 4))
digitImIxSet = np.zeros((10, 28000), dtype = int)

# REPORT FREQUENCY OF EACH IMAGE (AND TOTAL)
digitImFreq = np.zeros(10, dtype = int)
IMAGE_FREQ = 0

print("\nPreprocessing images...")

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
            if IMAGE_FREQ in digitIndexSet[digit]:
                add_digit(digit, smallPic, digitImSet, digitImIxSet, digitImFreq, IMAGE_FREQ)
                break

        IMAGE_FREQ = IMAGE_FREQ + 1
        bar()

# INVESTIGATE SAMPLES FROM APPROX 1% TO APPROX 20% OF IMAGES
Tset = [280, 560, 840, 1120, 1400, 1750, 2100, 2800, 3500, 4200, 4900, 5600]
ES = len(Tset)
T_FREQ = 0

for T in Tset:

    # STORE T IMAGES CORRESPONDING TO EACH DIGIT
    sampleImSet = np.zeros((10, T, 4, 4))
    sampleImList = np.zeros((10*T, 4, 4))
    sizeUniqueImSet = np.zeros(10)
    OVERALL_FREQ = 0

    print("\nFinding unique images...")

    for D in range(0, 10):

        # RANDOMLY SAMPLE T INDICES FROM EACH DIGIT SET
        randomIndices = random.sample(range(0, 28000), T)
        SAMPLE_FREQ = 0

        for index in randomIndices:

            # EXTRACT EACH IMAGE CORRESPONDING TO EACH OF THE T INDICES AND SAVE IN NEW STRUCTURE
            randomImage = digitImSet[D, index]
            sampleImSet[D, SAMPLE_FREQ] = randomImage
            sampleImList[OVERALL_FREQ] = randomImage
            SAMPLE_FREQ = SAMPLE_FREQ + 1
            OVERALL_FREQ = OVERALL_FREQ + 1

        # FIND FREQUENCIES OF ALL UNIQUE IMAGES IN SAMPLE IMAGE SET
        uniqueImSet = np.unique(sampleImSet[D], axis = 0)
        sizeUniqueImSet[D] = len(uniqueImSet)

    # FIND FREQUENCIES OF UNIQUE IMAGES IN SAMPLE IMAGE LIST
    uniqueImList = np.unique(sampleImList, axis = 0)

    # DOMAIN FOR EACH DIGIT DISTRIBUTION IS NUMBER OF UNIQUE IMAGES
    U = len(uniqueImList)

    # STORE FREQUENCIES OF UNIQUE IMAGES FOR EACH DIGIT
    uImageSet = np.zeros((10, U, 4, 4))
    uFreqSet = np.zeros((10, U))
    uProbsSet = np.zeros((10, U))

    print("Creating probability distributions...")

    # SMOOTHING PARAMETER: 0.1 AND 1 ARE TOO LARGE
    ALPHA = 0.01

    for D in range(0, 10):
        UNIQUE_FREQ = 0

        # STORE IMAGE AND SMOOTHED PROBABILITY AS WELL AS FREQUENCY
        for image in uniqueImList:
            where = np.where(np.all(image == sampleImSet[D], axis = (1, 2)))
            freq = len(where[0])
            uImageSet[D, UNIQUE_FREQ] = image
            uFreqSet[D, UNIQUE_FREQ] = int(freq)
            uProbsSet[D, UNIQUE_FREQ] = float((freq + ALPHA)/(T + (ALPHA*(sizeUniqueImSet[D]))))
            UNIQUE_FREQ = UNIQUE_FREQ + 1

    # FOR K3 ESTIMATOR (SCHULMAN) TAKE A SMALL SAMPLE OF UNIQUE IMAGES
    E = 10

    # STORE IMAGES, FREQUENCIES AND PROBABILITIES FOR THIS SUBSET
    eImageSet = np.ones((10, E, 4, 4))
    eFreqSet = np.zeros((10, E))
    eProbsSet = np.zeros((10, E))
    eTotalFreq = np.zeros(10)

    uSampledSet = np.random.choice(U, E, replace = False)

    # BORROW DATA FROM CORRESPONDING INDICES OF MAIN IMAGE AND FREQUENCY SETS
    for D in range(0, 10):
        for i in range(E):
            eImageSet[D, i] = uImageSet[D, uSampledSet[i]]
            eFreqSet[D, i] = uFreqSet[D, uSampledSet[i]]
            eTotalFreq[D] = sum(eFreqSet[D])
            eProbsSet[D, i] = float((eFreqSet[D, i] + ALPHA)/(T + (ALPHA*(eTotalFreq[D]))))

    # PARAMETERS FOR THE ADDITION OF LAPLACE AND GAUSSIAN NOISE
    EPS = 0.1
    DTA = 0.1
    A = 0
    R = 10

    # LIST OF THE TRIALS THAT WILL BE RUN
    trialset = ["end_lap", "end_lap_mc", "end_gauss", "end_gauss_mc", "mid_gauss", "mid_gauss_mc"]
    TS = len(trialset)

    # CONSTANTS FOR LAMBDA SEARCH
    rLda = 1
    ldaStep = 0.05
    L = int(rLda / ldaStep)

    # STORES FOR MEAN OF UNBIASED ESTIMATOR AND LAMBDA
    uEst = np.zeros((TS, ES, L))
    meanEst = np.zeros((TS, ES))

    for trial in range(6):
        print(f"\nTrial {trial + 1}: {trialset[trial]}")
        print(f"T = {T}: trial {trial + 1}...")

        # STORES FOR EXACT AND NOISY UNKNOWN DISTRIBUTIONS
        uDist = np.zeros((10, 10, U))
        nDist = np.zeros((10, 10, E))
        uList = []
        uCDList = []

        # STORES FOR RATIO BETWEEN UNKNOWN AND KNOWN DISTRIBUTIONS
        rList = []
        kList = []

        # OPTION 1A: BASELINE CASE
        if trial % 2 == 0:
            b1 = log(2) / EPS

        # OPTION 1B: MONTE CARLO ESTIMATE
        else:
            b1 = (1 + log(2)) / EPS

        b2 = (2*((log(1.25))/DTA)*b1) / EPS

        # FOR EACH COMPARISON DIGIT COMPUTE EXACT AND NOISY UNKNOWN DISTRIBUTIONS FOR ALL DIGITS
        for C in range(0, 10):
            for D in range(0, 10):

                for i in range(0, U):
                    uDist[C, D, i] = uProbsSet[D, i] * (np.log((uProbsSet[D, i]) / (uProbsSet[C, i])))

                for j in range(0, E):
                    nDist[C, D, j] = eProbsSet[D, j] * (np.log((eProbsSet[D, j]) / (eProbsSet[C, j])))

                # ELIMINATE ALL ZERO VALUES WHEN DIGITS ARE IDENTICAL
                if sum(uDist[C, D]) != 0.0:
                    uList.append(sum(uDist[C, D]))
                    uCDList.append((C, D))

                # COMPUTE RATIO BETWEEN EXACT AND NOISY UNKNOWN DISTRIBUTIONS
                ratio = abs(sum(nDist[C, D, j]) / sum(uDist[C, D]))

                # ELIMINATE ALL DIVIDE BY ZERO ERRORS
                if ratio != 0.0 and sum(uDist[C, D]) != 0.0:
                    rList.append(ratio)

                    # COMPUTE KNOWN DISTRIBUTION
                    kDist = abs(sum(nDist[C, D, j]) * log(ratio))
                    kList.append(kDist)

                    # WAIT UNTIL FINAL DIGIT PAIR (9, 8) TO ANALYSE EXACT UNKNOWN DISTRIBUTION LIST
                    if C == 9 and D == 8:
                        
                        # LOAD GAUSSIAN NOISE DISTRIBUTIONS FOR INTERMEDIATE SERVER
                        if trial >= 4:
                            s1 = b2 * (np.sqrt(2) / T)
                            s2 = b2 * lda * (np.sqrt(2) / T)
                            noise1 = tfp.distributions.Normal(loc = A, scale = s1)
                            noise2 = tfp.distributions.Normal(loc = A, scale = s2)

                        for row in range(0, len(rList)):
                            uLogr = log(rList[row])

                            # OPTION 2A: INTERMEDIATE SERVER ADDS NOISE TERMS
                            if trial >= 4:
                                uNoise1 = log(rList[row]) + noise1.sample(sample_shape = (1,))
                            
                            LDA_FREQ = 0

                            # EXPLORE LAMBDAS IN A RANGE
                            for lda in range(0, rLda, ldaStep):

                                # COMPUTE K3 ESTIMATOR
                                if trial >= 4:
                                    uNoise2 = uLogr.exp() + noise2.sample(sample_shape = (1,))
                                    uRangeEst = (lda * (uNoise2 - 1)) - uNoise1
                                
                                # OPTION 2B: NO NOISE UNTIL END
                                else:
                                    uRangeEst = lda * (uLogr.exp() - 1) - uLogr

                                # COMPARE UNKNOWN DISTRIBUTION ESTIMATOR TO KNOWN DISTRIBUTION
                                uEst[trial, T_FREQ, LDA_FREQ] = abs(uRangeEst - kList[row])
                                LDA_FREQ = LDA_FREQ + 1 

            uDict = dict(zip(uList, uCDList))
            oUDict = OrderedDict(sorted(uDict.items()))
            
            # UNKNOWN DISTRIBUTION IS IDENTICAL FOR ALL TS AND TRIALS
            if T == 280 and trial == 0:
                orderfile = open("emnist_unknown_dist_in_order.txt", "w", encoding = 'utf-8')
                orderfile.write("EMNIST: Unknown Distribution In Order\n")
                orderfile.write("Smaller corresponds to more similar digits\n\n")

                for i in oUDict:
                    orderfile.write(f"{i} : {oUDict[i]}\n")

        meanLda = np.zeros((L, ES))

        # COMPUTE MEAN ERROR OF UNBIASED ESTIMATOR FOR EACH LAMBDA
        for l in range(0, rLda, ldaStep):
            meanLda[l] = np.mean(uEst, axis = (0, 1))

        # FIND LAMBDA THAT PRODUCES MINIMUM ERROR
        meanOpt = np.mean(meanLda, axis = 1)
        meanIndex = np.argmin(meanOpt)
        ldaIndex = ldaStep * meanIndex

        # MEAN ACROSS CLIENTS FOR OPTIMUM LAMBDA
        meanEst = meanLda[ldaIndex]

        # OPTION 2B: SERVER ADDS NOISE TERM TO FINAL RESULT
        if trial < 4:
            
            # OPTION 3A: ADD LAPLACE NOISE
            if trial < 2:    
                endNoise = tfp.distributions.Laplace(loc = A, scale = s1)
            
            # OPTION 3B: ADD GAUSSIAN NOISE
            else:
                endNoise = tfp.distributions.Normal(loc = A, scale = s1)

            meanEst = meanEst + endNoise.sample(sample_shape = (1,))

        statsfile = open(f"emnist_{trialset[trial]}_noise_t_{T}.txt", "w", encoding = 'utf-8')
        statsfile.write(f"EMNIST: T = {T}\n")
        statsfile.write(f"Optimal Lambda {round(meanEst[trial, T_FREQ], 4)} for Mean {round(meanEst[trial, T_FREQ], 4)}\n\n")

        T_FREQ = T_FREQ + 1

# PLOT MEAN ERROR OF UNBIASED ESTIMATOR FOR EACH T
print(f"\nmeanEst: {meanEst[0]}")
plt.errorbar(Tset, meanEst[0], yerr = np.std(meanEst[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, meanEst[1], yerr = np.std(meanEst[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, meanEst[2], yerr = np.std(meanEst[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, meanEst[3], yerr = np.std(meanEst[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, meanEst[4], yerr = np.std(meanEst[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, meanEst[5], yerr = np.std(meanEst[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.xlabel("Value of T")
plt.ylabel("Error of unbiased estimator (mean)")
plt.title("How T affects error of unbiased estimator (mean)")
plt.savefig("Emnist_pixel_T_mean.png")
plt.clf()

# PLOT LAMBDAS FOR EACH T
print(f"meanLda: {meanLda[0]}")
plt.errorbar(Tset, meanLda[0], yerr = np.std(meanLda[0], axis = 0), color = 'tab:blue', marker = 'o', label = 'end lap')
plt.errorbar(Tset, meanLda[1], yerr = np.std(meanLda[1], axis = 0), color = 'tab:cyan', marker = 'x', label = 'end lap mc')
plt.errorbar(Tset, meanLda[2], yerr = np.std(meanLda[2], axis = 0), color = 'tab:olive', marker = 'o', label = 'end gauss')
plt.errorbar(Tset, meanLda[3], yerr = np.std(meanLda[3], axis = 0), color = 'tab:green', marker = 'x', label = 'end gauss mc')
plt.errorbar(Tset, meanLda[4], yerr = np.std(meanLda[4], axis = 0), color = 'tab:red', marker = 'o', label = 'mid gauss')
plt.errorbar(Tset, meanLda[5], yerr = np.std(meanLda[5], axis = 0), color = 'tab:pink', marker = 'x', label = 'mid gauss mc')
plt.legend(loc = 'best')
plt.yscale('log')
plt.xlabel("Value of T")
plt.ylabel("Lambda to minimise error of unbiased estimator")
plt.title("How T affects lambda (mean)")
plt.savefig("Emnist_pixel_T_lda.png")
plt.clf()

# COMPUTE TOTAL RUNTIME IN MINUTES AND SECONDS
totalTime = time.perf_counter() - startTime

if (totalTime // 60) == 1:
    print(f"Runtime: {round(totalTime // 60)} minute {round((totalTime % 60), 2)} seconds.\n")
else:
    print(f"Runtime: {round(totalTime // 60)} minutes {round((totalTime % 60), 2)} seconds.\n")
