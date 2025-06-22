import soundfile as sf
from pystoi import stoi

def score(clean, denoised, fs, extended=False):
    """
    Calculate the Short-Time Objective Intelligibility (STOI) score.
    
    Parameters:
    clean (str): Path to the clean reference audio file.
    denoised (str): Path to the degraded audio file.
    fs (int): Sampling frequency of the audio files.
    extended (bool): If True, use extended STOI calculation.
    
    Returns:
    float: STOI score between 0 and 1.
    """
    
    # Read the audio files
    clean, fs = sf.read('path/to/clean/audio')
    denoised, fs = sf.read('path/to/denoised/audio')

    # Clean and den should have the same length, and be 1D
    d = stoi(clean, denoised, fs, extended=False)

    return d