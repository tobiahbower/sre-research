import pywt
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load speech waveform (mono) using librosa
y, sr = librosa.load("speech.wav", sr=None)  # Use your WAV file here

# Choose a wavelet (e.g., Daubechies 4)
wavelet = 'db4'

# Perform multilevel discrete wavelet decomposition
coeffs = pywt.wavedec(y, wavelet, level=5)

# Plot original signal
plt.figure(figsize=(12, 8))
plt.subplot(len(coeffs)+1, 1, 1)
plt.plot(y)
plt.title("Original Speech Waveform")

# Plot wavelet coefficients
for i, coeff in enumerate(coeffs):
    plt.subplot(len(coeffs)+1, 1, i+2)
    plt.plot(coeff)
    if i == 0:
        plt.title("Approximation Coefficients (Level 5)")
    else:
        plt.title(f"Detail Coefficients (Level {5 - i + 1})")

plt.tight_layout()
plt.show()

## pip install pywavelets