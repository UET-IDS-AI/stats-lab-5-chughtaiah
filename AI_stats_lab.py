import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    PDF of exponential distribution.

    f(x) = lam * exp(-lam*x)  for x >= 0
    f(x) = 0                  for x < 0
    """
    if x < 0:
        return 0.0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Analytical probability P(a < X < b)
    using CDF of exponential distribution.

    F(x) = 1 - exp(-lam*x)
    """
    Fa = 1 - np.exp(-lam * a)
    Fb = 1 - np.exp(-lam * b)

    return Fb - Fa


def simulate_exponential_probability(a, b, n=100000):
    """
    Monte-Carlo simulation estimate
    of P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)

    count = np.sum((samples > a) & (samples < b))

    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Gaussian probability density function.

    f(x) =
    1/(sqrt(2*pi)*sigma) * exp(-(x-mu)^2 / (2*sigma^2))
    """
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)

    return coeff * np.exp(exponent)


def posterior_probability(time):
    """
    Compute posterior probability P(B | X = time)
    using Bayes rule.

    Priors
    P(A) = 0.3
    P(B) = 0.7

    Distributions
    A ~ N(40, 4)  -> sigma = 2
    B ~ N(45, 4)  -> sigma = 2
    """

    # Priors
    pA = 0.3
    pB = 0.7

    # Standard deviation
    sigma = 2

    # Likelihoods
    fA = gaussian_pdf(time, 40, sigma)
    fB = gaussian_pdf(time, 45, sigma)

    # Bayes rule
    numerator = pB * fB
    denominator = pA * fA + pB * fB

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate posterior probability via simulation.
    """

    # Priors
    pA = 0.3
    pB = 0.7

    # Sample group labels
    groups = np.random.choice(["A", "B"], size=n, p=[pA, pB])

    # Generate finishing times
    times = np.zeros(n)

    for i in range(n):
        if groups[i] == "A":
            times[i] = np.random.normal(40, 2)
        else:
            times[i] = np.random.normal(45, 2)

    # Identify swimmers close to observed time
    tolerance = 0.5
    mask = np.abs(times - time) < tolerance

    if np.sum(mask) == 0:
        return 0

    selected_groups = groups[mask]

    prob_B = np.sum(selected_groups == "B") / len(selected_groups)

    return prob_B
