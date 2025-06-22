# Developed by O. Das in MATLAB, ported into Python by TJ. Bower

import numpy as np
from scipy.signal import lfilter

def aryule(x, order):
    """Estimate AR coefficients using Yule-Walker method."""
    r = np.correlate(x, x, mode='full')[-len(x):]
    R = np.array([r[i:i+order] for i in range(order)])
    b = -np.linalg.pinv(R) @ r[1:order+1]
    noise_var = r[0] + np.dot(b, r[1:order+1])
    return np.concatenate(([1], b)), noise_var, -b

def find_order(noisy):
    """Estimate order of each frame of noisy signal."""
    # print(noisy.shape[0])  # This will show the shape of the array
    samples_per_segment = 1000  # Adjust this value based on your needs
    noisy = noisy.reshape(-1, samples_per_segment)
    totseg = noisy.shape[0]
    order = np.zeros(totseg, dtype=int)
    T = 100  # Maximum order assumption

    for i in range(totseg):
        arcoefs, noisevar, reflection_coefs = aryule(noisy[i, :], T)
        pacf = reflection_coefs
        cpacf = np.cumsum(np.abs(pacf))
        dist = np.abs(cpacf - 0.7 * np.ptp(cpacf))
        order[i] = np.argmin(dist) + 1

    return order