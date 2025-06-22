import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def load_audio(filepath, sr=22050, offset=0.0, duration=None):
    """ Load an audio file and return the waveform and sample rate. """
    y, fs = librosa.load(filepath, sr=sr, offset=offset, duration=duration)
    print("Audio input fs:", fs, "Hz\n")
    return y, fs

def compute_spectrogram(y, fs, plot=False, output_path=None):
    """ Compute and optionally plot a power spectrogram. """
    y_stft = np.abs(librosa.stft(y))
    y_stft_db = librosa.amplitude_to_db(y_stft, ref=np.max)
    
    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(y_stft_db, x_axis='time', y_axis='log', sr=fs)
        plt.colorbar(format='%+2.0f dB')
        plt.title("Power Spectrogram")
        plt.show(block=False)
        if output_path:
            plt.savefig(output_path, dpi=300)

def process_audio(audio_data_or_path, sr, pwrThd=-10, windowSize=2048):
    if isinstance(audio_data_or_path, str):
        # If given a path, load the file
        y, sr = librosa.load(audio_data_or_path, sr=sr)
    else:
        # Assume it's already a NumPy array
        y = audio_data_or_path

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=windowSize, hop_length=windowSize // 4,
        power=2.0
    )
    # Continue rest of processing...
    y_gla = librosa.griffinlim(mel, n_iter=2)
    y_istft = librosa.istft(librosa.stft(y))
    
    return mel, y_gla, y_istft


def save_audio(filename, y, fs):
    """ Save audio waveform to a file. """
    sf.write(filename, y, fs)

def plot_waveforms(y, y_gla, y_istft, fs, output_path=None):
    """ Plot and compare waveforms. """
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    librosa.display.waveshow(y, sr=fs, color='b', ax=ax[0])
    ax[0].set(title="Original Waveform")
    librosa.display.waveshow(y_gla, sr=fs, color='g', ax=ax[1])
    ax[1].set(title="Griffin-Lim Reconstruction")
    librosa.display.waveshow(y_istft, sr=fs, color='r', ax=ax[2])
    ax[2].set_title("Magnitude-only ISTFT Reconstruction")
    
    plt.show(block=False)
    if output_path:
        plt.savefig(output_path, dpi=300)

def plot_spectrograms(mel, mel_gla, mel_istft, fs, output_path=None):
    """ Plot and compare spectrograms. """
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    i1 = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=fs, cmap='viridis', ax=ax[0])
    fig.colorbar(i1, ax=[ax[0]], format='%+2.0f dB')
    ax[0].set(title="Mel Spectrogram")
    
    i2 = librosa.display.specshow(mel_gla, x_axis='time', y_axis='mel', sr=fs, cmap='viridis', ax=ax[1])
    fig.colorbar(i2, ax=[ax[1]], format='%+2.0f dB')
    ax[1].set(title="Griffin-Lim Reconstruction")
    
    i3 = librosa.display.specshow(mel_istft, x_axis='time', y_axis='mel', sr=fs, cmap='viridis', ax=ax[2])
    fig.colorbar(i3, ax=[ax[2]], format='%+2.0f dB')
    ax[2].set(title="Magnitude-only ISTFT Reconstruction")
    
    plt.show()
    if output_path:
        fig.savefig(output_path, dpi=300)

# # Example Usage
# filepath = './michelle-inputs/michelle.wav'
# y, fs = load_audio(filepath, sr=22050, offset=7.0, duration=20.0)
# compute_spectrogram(y, fs, plot=True, output_path='michelle-outputs/powerthd-10.png')

# mel, y_gla, y_istft = process_audio(y, fs, pwrThd=-10)
# save_audio('michelle_recon.wav', y, fs)
# save_audio('michelle_recon_gla.wav', y_gla, fs)
# save_audio('michelle_recon_istft.wav', y_istft, fs)

# plot_waveforms(y, y_gla, y_istft, fs, output_path='michelle-outputs/waveforms.png')
# plot_spectrograms(mel, librosa.feature.melspectrogram(y=y_gla, sr=fs, n_fft=2048, hop_length=512, 
#                   win_length=None, window='hann', center=True, pad_mode='reflect', power=1.0, n_mels=256), 
#                   librosa.feature.melspectrogram(y=y_istft, sr=fs, n_fft=2048, hop_length=512, 
#                   win_length=None, window='hann', center=True, pad_mode='reflect', power=1.0, n_mels=256), 
#                   fs, output_path='michelle-outputs/spectrograms-power1.png')