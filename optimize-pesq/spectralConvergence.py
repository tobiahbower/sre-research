import numpy as np

def is_hermitian(matrix):
    """
    Checks if a matrix is Hermitian.

    Args:
        matrix (numpy.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """
    return np.allclose(matrix, matrix.conj().T)

def check_hermitian_symmetry(y_gla):
    stft_y_gla = np.fft.rfft(y_gla)
    return np.allclose(np.imag(stft_y_gla), 0)

def calculate_spectral_convergence(errors):
    """
    Calculate spectral convergence using the formula:
    ||E_n||_∞ / ||E_0||_∞ = (1 + σ)^(-n)
    where σ is the spectral radius.

    Parameters:
    errors (list): List of error values at each iteration.

    Returns:
    float: Spectral convergence value.
    """
    # Calculate the maximum error at each iteration
    max_errors = [np.max(err) for err in errors]

    # Calculate the spectral convergence value
    spectral_convergence = np.inf
    for n in range(1, len(max_errors)):
        ratio = max_errors[n] / max_errors[0]
        if n == 1:
            spectral_convergence = ratio
        else:
            spectral_convergence = np.min([spectral_convergence, ratio])

    return spectral_convergence

def calculate_ssnr(y, y_gla):
    stft_y = np.abs(np.fft.rfft(y))
    stft_y_gla = np.abs(np.fft.rfft(y_gla))
    ssnr = 10 * np.log10(np.sum(stft_y**2) / np.sum((stft_y - stft_y_gla)**2))
    return ssnr