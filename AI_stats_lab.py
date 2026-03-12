import numpy as np

def exponential_pdf(x, lam=1):
    if x < 0:
        return 0
    return lam*np.exp(-lam*x)


def exponential_interval_probability(a=2,b=5,lam=1):
    return np.exp(-a) - np.exp(-b)


def simulate_exponential_probability(a=2,b=5,n=100000):
    samples = np.random.exponential(scale=1,size=n)
    return np.mean((samples>a)&(samples<b))


def gaussian_pdf(x,mu,sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))


def posterior_probability(time=42):

    pA=0.3
    pB=0.7
    sigma=2

    fA=gaussian_pdf(time,40,sigma)
    fB=gaussian_pdf(time,45,sigma)

    return (pB*fB)/(pA*fA+pB*fB)
