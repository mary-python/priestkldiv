"""Modules are for mathematical / statistical computations, and for graph plotting."""
from math import log
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

a, b1 = 0, 2*log(1.25)*log(2)
rv1 = norm(a, b1)

b2 = 2*(log(1.25)/(1/2))*log(2)
rv2 = norm(a, b2)

b3 = 2*(log(1.25)/2)*log(2)
rv3 = norm(a, b3)

b4 = 2*(log(1.25)/5)*log(2)
rv4 = norm(a, b4)

b5 = 2*(log(1.25)/(1/3))*log(2)
rv5 = norm(a, b5)

dist1 = np.linspace(-2, np.minimum(rv1.dist.b, 2))
dist2 = np.linspace(-2, np.minimum(rv2.dist.b, 2))
dist3 = np.linspace(-2, np.minimum(rv3.dist.b, 2))
dist4 = np.linspace(-2, np.minimum(rv4.dist.b, 2))
dist5 = np.linspace(-2, np.minimum(rv5.dist.b, 2))

plot4 = plt.plot(dist4, rv4.pdf(dist4), label = "\u03B4=0.2")
plot3 = plt.plot(dist3, rv3.pdf(dist3), label = "\u03B4=0.5")
plot1 = plt.plot(dist1, rv1.pdf(dist1), label = "\u03B4=1")
plot2 = plt.plot(dist2, rv2.pdf(dist2), label = "\u03B4=2")
plot5 = plt.plot(dist5, rv5.pdf(dist5), label = "\u03B4=3")

plt.ylim(-0.03, 5.4)
plt.legend(loc="upper left")
plt.show()
