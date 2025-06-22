from dataclasses import dataclass

import librosa
import mel as mel
import griffinLimTune as gla
import scorePESQ as pesqMetric
import scoreSTOI as stoi
import speechOrder as kalman
import cepstralPitch as cepstral

from scipy.signal import resample
import numpy as np
from pydub import AudioSegment

# @dataclass
# class structAudio22:
#     field1: str
#     field2: str
#     field3: str
#     field4: str

# # create struct of audio input files
# audio_files = structAudio22(
#     field1='./billy-inputs/billy.wav',
#     field2='./michelle-inputs/michelle.wav',
#     field3='./billy-inputs/billy_recon.wav',
#     field4='./michelle-inputs/michelle_recon.wav',
# )

# User Configuration
sr_pesq_req = 16000  # Sampling frequency
pwrThd = -10  # Power threshold for Mel Spectrogram
windowSizes = []
pesqScores = []
stoiScores = []
ssnrScores = []
fundamentalFrequencies = []
orders = []

# New contribution: Loop NOIZEUS to increase length (2 sec is not long enough)
# # Only focus on the 0dB segments of NOIZEUS Corpus
# noizeus_path = './../noizeus_corpora/NOIZEUS/'
# noisy_segments = []
# for dirpath, dirnames, filenames in os.walk(noizeus_path):
#     for dirname in dirnames:
#         if "0dB" in dirname:
#             noisy_segments.append(os.path.join(dirpath, dirname))

audio_files = [
    './billy-inputs/billy_recon.wav',
    './michelle-inputs/michelle_recon.wav',
]

noise_files = [
    './../db-demand/SPSQUARE/ch01.wav', # Street noise
    './../db-demand/SPSQUARE/ch02.wav',
    './../db-demand/SPSQUARE/ch03.wav',
    './../db-demand/SPSQUARE/ch04.wav',
    './../db-demand/SPSQUARE/ch05.wav',
    './../db-demand/SPSQUARE/ch06.wav',
    './../db-demand/SPSQUARE/ch07.wav',
    './../db-demand/SPSQUARE/ch08.wav',
    './../db-demand/SPSQUARE/ch09.wav',
    './../db-demand/SPSQUARE/ch10.wav',
    './../db-demand/SPSQUARE/ch11.wav',
    './../db-demand/SPSQUARE/ch12.wav',
    './../db-demand/SPSQUARE/ch13.wav',
    './../db-demand/SPSQUARE/ch14.wav',
    './../db-demand/SPSQUARE/ch15.wav',
    './../db-demand/STRAFFIC/ch01.wav',
    './../db-demand/STRAFFIC/ch02.wav',
    './../db-demand/STRAFFIC/ch03.wav',
    './../db-demand/STRAFFIC/ch04.wav',
    './../db-demand/STRAFFIC/ch05.wav',
    './../db-demand/STRAFFIC/ch06.wav',
    './../db-demand/STRAFFIC/ch07.wav',
    './../db-demand/STRAFFIC/ch08.wav',
    './../db-demand/STRAFFIC/ch09.wav',
    './../db-demand/STRAFFIC/ch10.wav',
    './../db-demand/STRAFFIC/ch11.wav',
    './../db-demand/STRAFFIC/ch12.wav',
    './../db-demand/STRAFFIC/ch13.wav',
    './../db-demand/STRAFFIC/ch14.wav',
    './../db-demand/STRAFFIC/ch15.wav',
    './../db-demand/TMETRO/ch01.wav', # Transportation noise
    './../db-demand/TMETRO/ch02.wav',
    './../db-demand/TMETRO/ch03.wav',
    './../db-demand/TMETRO/ch04.wav',
    './../db-demand/TMETRO/ch05.wav',
    './../db-demand/TMETRO/ch06.wav',
    './../db-demand/TMETRO/ch07.wav',
    './../db-demand/TMETRO/ch08.wav',
    './../db-demand/TMETRO/ch09.wav',
    './../db-demand/TMETRO/ch10.wav',
    './../db-demand/TMETRO/ch11.wav',
    './../db-demand/TMETRO/ch12.wav',
    './../db-demand/TMETRO/ch13.wav',
    './../db-demand/TMETRO/ch14.wav',
    './../db-demand/TMETRO/ch15.wav',
    
]

# billy, fs = mel.load_audio(audio_files[0], sr=22050, offset=0.0, duration=20.0)
# michelle, fs = mel.load_audio(audio_files[1], sr=22050, offset=0.0, duration=20.0)

for wavfile in noise_files:
    # Iterate through input files
    eta, fs = mel.load_audio(noise_files[0], sr=22050, offset=0.0, duration=20.0)
    
    sound1 = AudioSegment.from_file(audio_files[0])
    sound2 = AudioSegment.from_file(audio_files[1])
    combined1 = sound1.overlay(eta, position=0) # ms
    combined2 = sound2.overlay(eta, position=0) # ms

    # Export the combined audio
    # combined1.export("combined_audio1.wav", format="wav")
    # combined2.export("combined_audio2.wav", format="wav")


# Iterate through window sizes 32-65536 (2 ms - ~4 sec)
for windowSize in range(32, 64536, 32):

    windowSizes.append(windowSize) # for plotting

    # Built-in Reconstruction: Griffin-Lim vs. ISTFT
    mel, y_gla, y_istft = mel.process_audio(combined1, fs, pwrThd=pwrThd, windowSize=windowSize)

    # Custom Griffin-Lim Reconstruction (only 2 iterations, not optimized to avoid introducing complexity)
    y = resample(x, fs, 16000)
    D = gla.compute_stft(y)
    magnitude = np.abs(D)
    y_gla = gla.griffin_lim(magnitude, n_iter=2, alpha=0.99, lambda_=0.1)

    # Estimate the order of the speech signal using Kalman filter
    order = kalman.find_order(y_gla)
    orders = orders.append(order)

    # Kalman Speech Enhancement
    clean, denoised = kalman.cmi_tuned_kalman_speech(y_gla, fs)

    # Calculate PESQ Score
    score = pesqMetric.score(sr_pesq_req, Xr, Yr, mode='wb')
    pesqScores.append(score)

    # Calculate STOI Score
    stoi = stoi.score(clean, denoised, fs, extended=False)
    stoiScores.append(stoi)

    # Calculate SSNR
    ssnr = gla.calculate_ssnr(y, y_gla)
    ssnrScores.append(ssnr)

    # Calculate cepstral pitch (fundamental frequency) of reconstructed audio
    pitch = cepstral.get_pitch()
    fundamentalFrequencies.append(pitch)

# Plot SSNR