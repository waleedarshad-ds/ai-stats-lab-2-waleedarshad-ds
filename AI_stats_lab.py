import math
import numpy as np
import matplotlib.pyplot as plt

# Part 1 — Probability Foundations
def probability_union(PA, PB, PAB):
    return PA + PB - PAB


def conditional_probability(PAB, PB):
    if PB == 0:
        return 0
    return PAB / PB

def are_independent(PA, PB, PAB, tol=1e-9):
    return abs(PAB - (PA * PB)) < tol
def bayes_rule(PBA, PA, PB):
    if PB == 0:
        return 0
    return (PBA * PA) / PB

# Part 2 — Bernoulli Distribution

def bernoulli_pmf(x, theta):
    return (theta ** x) * ((1 - theta) ** (1 - x))

def bernoulli_theta_analysis(theta_values):
    results = []
    for theta in theta_values:
        P0 = bernoulli_pmf(0, theta)
        P1 = bernoulli_pmf(1, theta)
        is_symmetric = abs(theta - 0.5) < 1e-9
        results.append((theta, P0, P1, is_symmetric))
    return results

# Part 3 — Normal Distribution

def normal_pdf(x, mu, sigma):
    if sigma == 0:
        return 0
    coeff = 1 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coeff * math.exp(exponent)


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):
    results = []

    for mu in mu_values:
        for sigma in sigma_values:
            samples = np.random.normal(mu, sigma, n_samples)

            sample_mean = np.mean(samples)
            theoretical_mean = mu
            mean_error = abs(sample_mean - theoretical_mean)

            sample_variance = np.var(samples)
            theoretical_variance = sigma ** 2
            variance_error = abs(sample_variance - theoretical_variance)

            # Histogram 
            plt.hist(samples, bins=bins, density=True)
            plt.title(f"Normal Distribution μ={mu}, σ={sigma}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.close()

            results.append((
                mu,
                sigma,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            ))

    return results

# Part 4 — Uniform Distribution

def uniform_mean(a, b):
    return (a + b) / 2

def uniform_variance(a, b):
    return ((b - a) ** 2) / 12

def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30):
    results = []

    for a in a_values:
        for b in b_values:
            samples = np.random.uniform(a, b, n_samples)

            sample_mean = np.mean(samples)
            theoretical_mean = uniform_mean(a, b)
            mean_error = abs(sample_mean - theoretical_mean)

            sample_variance = np.var(samples)
            theoretical_variance = uniform_variance(a, b)
            variance_error = abs(sample_variance - theoretical_variance)

            # Histogram
            plt.hist(samples, bins=bins, density=True)
            plt.title(f"Uniform Distribution a={a}, b={b}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.close()

            results.append((
                a,
                b,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            ))

    return results


if __name__ == "__main__":
    print("All functions implemented successfully.")
