"""Modules are for mathematical / statistical computations, and for graph plotting."""
from math import log
from scipy.stats import laplace
import numpy as np
import matplotlib.pyplot as plt

a, b1 = 0, (1+log(2))
rv1 = laplace(a, b1)

b2 = (1+log(2))/2
rv2 = laplace(a, b2)

b3 = 2*(1+log(2))
rv3 = laplace(a, b3)

b4 = 5*(1+log(2))
rv4 = laplace(a, b4)

b5 = (1+log(2))/3
rv5 = laplace(a, b5)

dist1 = np.linspace(-2, np.minimum(rv1.dist.b, 2))
dist2 = np.linspace(-2, np.minimum(rv2.dist.b, 2))
dist3 = np.linspace(-2, np.minimum(rv3.dist.b, 2))
dist4 = np.linspace(-2, np.minimum(rv4.dist.b, 2))
dist5 = np.linspace(-2, np.minimum(rv5.dist.b, 2))

plot4 = plt.plot(dist4, rv4.pdf(dist4), label = "\u03B5=0.2")
plot3 = plt.plot(dist3, rv3.pdf(dist3), label = "\u03B5=0.5")
plot1 = plt.plot(dist1, rv1.pdf(dist1), label = "\u03B5=1")
plot2 = plt.plot(dist2, rv2.pdf(dist2), label = "\u03B5=2")
plot5 = plt.plot(dist5, rv5.pdf(dist5), label = "\u03B5=3")

plt.ylim(-0.01, 1.90)
plt.legend(loc="upper left")
plt.show()
