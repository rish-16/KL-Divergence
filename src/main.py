import math
import numpy as np
from scipy.stats import norm
from scipy.special import kl_div
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

mu1 = 10
variance1 = 2
sigma1 = math.sqrt(variance1)

mu2 = 10
variance2 = 2
sigma2 = math.sqrt(variance2)

p = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 50)
q = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 50)
score = np.sum(kl_div(p, q))

# enable math text
plt.rcParams.update({'mathtext.default': 'regular'})
plt.plot(p, norm.pdf(p, mu1, sigma1), color="coral", label="p")
plt.plot(q, norm.pdf(q, mu2, sigma2), color="#feca57", label="q")
plt.title("KL(p||q) = {:.2f} for \n $μ_p$={:.2f}|$σ_p$={:.2f} and $μ_q$={:.2f}|$σ_q$={:.2f}".format(score, mu1, sigma1, mu2, sigma2))
plt.legend()
plt.show()
