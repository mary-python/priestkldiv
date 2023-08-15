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
evenImageIndexSet = np.zeros(140000)
oddImageIndexSet = np.zeros(140000)

# KEEP TRACK OF HOW MANY ODD/EVEN/TOTAL IMAGES ARE PROCESSED
evenImageCount = 0
oddImageCount = 0
imageCount = 0

for pic in images:

    # IF ASSOCIATED WITH AN EVEN LABEL THEN ADD TO EVEN IMAGE SET
    if imageCount in evenIndexSet:
        evenImageSet[evenImageCount] = pic
        evenImageIndexSet[evenImageCount] = imageCount
        evenImageCount = evenImageCount + 1

    # IF ASSOCIATED WITH AN ODD LABEL THEN ADD TO ODD IMAGE SET
    else:
        oddImageSet[oddImageCount] = pic
        oddImageIndexSet[oddImageCount] = imageCount
        oddImageCount = oddImageCount + 1

    imageCount = imageCount + 1

# INSERT SCHULMAN TSAMPLE EPS CODE BELOW
import torch
torch.manual_seed(12)
import torch.distributions as dis
from math import log
import tensorflow as tf
tf.random.set_seed(638)
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

p = dis.Laplace(loc=0.1, scale=1)
q = dis.Normal(loc=0, scale=1)
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# sample T points
T = 10_000

# round to 2 d.p., find indices of and eliminate unique values
qSample = q.sample(sample_shape=(T,))
qRound = torch.round(qSample, decimals=2)
qUnique = torch.unique(qRound, return_counts=True)
qIndices = (qUnique[1] == 1).nonzero().flatten()
qUniqueIndices = qUnique[0][qIndices]

for i in qUniqueIndices:
    qRound = qRound[qRound != i]

qT = torch.numel(qRound)

# INSERT CODE HERE
# What kind of images do I want?
# 3D (10000, 28, 28) or vector (7.84M)?
# Use qRound indices to sample from oddImageSet

# skip Prio step for now
logr = (p.log_prob(qRound) - q.log_prob(qRound))
k3 = ((logr.exp() - 1) - logr)

# add Laplace and Gaussian noise with parameter(s) eps (and dta)
epsset = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4]
dta = 0.1
a = 0
b1 = log(2)
b2 = 2*((log(1.25))/dta)*b1

noise1 = tfp.distributions.Laplace(loc=a, scale=b1)
noise2 = tfp.distributions.Normal(loc=a, scale=b2)
KLDest1 = list()
KLDest2 = list()

for eps in epsset:

    k3noise1 = list()
    k3noise2 = list()

    for j in range(0, 10):
        k3noise1.append(k3 + (noise1.sample(sample_shape=qT, seed=12))/eps)
        k3noise2.append(k3 + (noise2.sample(sample_shape=qT, seed=12))/eps)

    average1 = sum(k3noise1) / len(k3noise1)
    average2 = sum(k3noise2) / len(k3noise2)
    KLDest1.append(abs(np.mean(average1) - truekl))
    KLDest2.append(abs(np.mean(average2) - truekl))

plot1 = plt.plot(epsset, KLDest1, label = f"Laplace dist")
plot2 = plt.plot(epsset, KLDest2, label = f"Gaussian dist")

plt.title("Effect of epsilon on noisy estimate of KLD")
plt.xlabel("Value of epsilon")
plt.ylabel("Difference in KLD")
plt.legend(loc="best")
plt.show()