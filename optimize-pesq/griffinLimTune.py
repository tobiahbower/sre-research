import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

pwrThd = -10

def load_audio(filepath, sr=16000):
    """ Load an audio file and return the waveform and sample rate. """
    y, sr = librosa.load(filepath, sr=sr)
    return y, sr

def compute_stft(y):
    """ Compute the Short-Time Fourier Transform (STFT). """
    return librosa.stft(y)

def griffin_lim(magnitude, n_iter=2, alpha=0.99, lambda_=0.1):
    """ Custom Griffin-Lim reconstruction with alpha and lambda adjustments. """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    S = magnitude * phase

    for _ in range(n_iter):
        y = librosa.istft(S)
        S = librosa.stft(y)
        S = magnitude * np.exp(1j * np.angle(S))
        S = (1 / (1 + lambda_)) * (S + (lambda_ / (1 + lambda_)) * magnitude * np.exp(1j * np.angle(S)))

    return librosa.istft(S)

def save_audio(filename, y, sr):
    """ Save reconstructed audio. """
    sf.write(filename, y, sr)

def plot_waveform(y, sr, output_path):
    """ Plot and save waveform. """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.savefig(output_path, dpi=300)
    plt.show()

def compute_mel_spectrogram(y, sr, pwrThd):
    """ Compute mel spectrogram. """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=1.0, n_mels=256
    )
    mel[mel < pwrThd] = pwrThd
    return mel

def plot_spectrogram(mel, sr, output_path):
    """ Plot and save mel spectrogram. """
    plt.figure(figsize=(10, 2))
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(output_path, dpi=300)
    plt.show()

# Example Usage
# filepath = './billy_recon.wav'
# y, sr = load_audio(filepath, sr=16000)
# D = compute_stft(y)
# magnitude = np.abs(D)

# reconstructed_audio = griffin_lim(magnitude, n_iter=2, alpha=0.99, lambda_=0.01)
# save_audio('reconstructed_billy_custom.wav', reconstructed_audio, sr)
# plot_waveform(reconstructed_audio, sr, "billy-outputs/iterated-waveform.png")

# mel = compute_mel_spectrogram(reconstructed_audio, sr, pwrThd)
# plot_spectrogram(mel, sr, "billy-outputs/iterated-spectrogram.png")