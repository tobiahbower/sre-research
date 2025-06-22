'''
# A good PESQ score generally ranges from 3.30 to 4.50. 
# Here’s how these scores are typically interpreted:

# 3.30 – 3.79: Attention necessary, but no appreciable effort required. The audio quality is acceptable, but there might be some minor issues that require slight attention.
# 3.80 – 4.50: Complete relaxation possible; no effort required. This range indicates excellent audio quality, where the conversation is clear and effortless for both parties.
# Scores below 3.30 are generally considered poor and can lead to significant frustration and communication difficulties:

# 2.80 – 3.29: Attention necessary; a small amount of effort required. The audio quality is subpar, and listeners may need to ask for repetitions.
# 1.00 – 1.99: No meaning understood with any feasible effort. The audio quality is very poor, making it nearly impossible to understand the conversation.
'''
import pypesq
from scipy.io import wavfile
from scipy.signal import resample

'''
# Calculates the PESQ score of a reference wav file and a degraded wav file.
    # Example usage:
    # ref = 'billy.wav'
    # deg = 'billy_recon.wav'
    # fs = 16000
    # wideband (50-7000 Hz)
    # narrowband (300-3400 Hz)
'''
def score(ref, deg, fs, wideband=True):

    sr, X = wavfile.read(ref)
    sr, Y = wavfile.read(deg)

    Xs = int(len(X) * fs / sr)
    Ys = int(len(Y) * fs / sr)

    Xr = resample(X, Xs)
    Yr = resample(Y, Ys)

    if wideband:
        score = pypesq.pesq(fs, Xr, Yr, mode='wb')
    else:
        score = pypesq.pesq(fs, Xr, Yr, mode='nb')
    
    print('PESQ:', score)

    return score