from dataclasses import dataclass
import librosa
import mel as mel
import griffinLimTune as gla
import scorePESQ as pesqMetric
import scoreSTOI as stoiMetric
import speechOrder as yule
import kalmanFilter as kalman
import cepstralPitch as cepstral

from scipy.signal import resample
import numpy as np
from pydub import AudioSegment
import os

import warnings
warnings.filterwarnings("ignore")

# import subprocess
# subprocess.run(["ffmpeg", "-i", "input.mp4", "-loglevel", "quiet", "output.mp4"])

os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# User Configuration
sr_target = 16000  # Target sampling frequency
pwrThd = -10  # Power threshold for Mel Spectrogram
windowSizes = []
pesqScores = []
stoiScores = []
ssnrScores = []
fundamentalFrequencies = []
orders = []

audio_files = [
    './billy-inputs/billy_recon.wav',
    './michelle-inputs/michelle_recon.wav',
]

noise_files = [
    './../db-demand/SPSQUARE/ch01.wav',
]

for wavfile in noise_files:
    # Load noise (only using the first for now)
    eta, fs = mel.load_audio(wavfile, sr=22050, offset=0.0, duration=20.0)

    # Load original audio files
    sound1 = AudioSegment.from_file(audio_files[0])
    sound2 = AudioSegment.from_file(audio_files[1])

    # Convert noise numpy array to AudioSegment
    eta_pcm = (eta * 32767).astype(np.int16).tobytes()
    eta_segment = AudioSegment(
        data=eta_pcm,
        sample_width=2,
        frame_rate=fs,
        channels=1
    )

    # Overlay noise on speech
    combined1 = sound1.overlay(eta_segment, position=0)
    combined2 = sound2.overlay(eta_segment, position=0)

    # Export or convert to numpy
    combined1.export("temp_combined1.wav", format="wav")
    combined2.export("temp_combined2.wav", format="wav")
    x, fs = librosa.load("temp_combined1.wav", sr=None)

    for windowSize in range(32, 64536, 32):
        windowSizes.append(windowSize)

        # Mel + Griffin-Lim + ISTFT
        melSpec, y_gla_init, y_istft = mel.process_audio("temp_combined1.wav", fs, pwrThd=pwrThd, windowSize=windowSize)

        # Custom Griffin-Lim Reconstruction
        num_samples = int(len(x) * sr_target / fs)
        y_resampled = resample(x, num_samples)
        D = gla.compute_stft(y_resampled)
        magnitude = np.abs(D)
        y_gla = gla.griffin_lim(magnitude, n_iter=2, alpha=0.99, lambda_=0.1)

        # Order estimation using Yule Walker method
        order = yule.find_order(y_gla)
        orders.append(order)

        # Kalman Speech Enhancement
        clean, denoised = kalman.cmi_tuned_kalman_speech(y_gla, sr_target)

        # PESQ
        score = pesqMetric.score(sr_target, clean, denoised, mode='wb')
        pesqScores.append(score)

        # STOI
        score_stoi = stoiMetric.score(clean, denoised, sr_target, extended=False)
        stoiScores.append(score_stoi)

        # SSNR
        ssnr = gla.calculate_ssnr(y_resampled, y_gla)
        ssnrScores.append(ssnr)

        # Cepstral pitch
        pitch = cepstral.get_pitch(y_gla)
        fundamentalFrequencies.append(pitch)

# Plotting or saving results can go here.

clear_screen()
