import numpy as np
from scipy.signal import lfilter, correlate
from scipy.fft import fft, fftshift
from scipy.stats import gmean

def measurement_noise_new(xseg, fs):
    num_frames = xseg.shape[0]
    noise_cov = np.zeros(num_frames)
    spectral_flatness = np.zeros(num_frames)
    silent_inds = np.zeros(num_frames)

    for k in range(num_frames):
        c = correlate(xseg[k, :], xseg[k, :], mode="full", method="direct") / np.var(xseg[k, :])
        psd = fftshift(np.abs(fft(c)))
        psd = psd[len(psd) // 2:]
        # freq = fs * np.arange(len(c) // 2) / len(c)
        freq = fs * np.arange(len(psd)) / len(c)

        #print(f"psd shape: {psd.shape}, freq shape: {freq.shape}")

        freq = freq[:len(psd)]  # Ensure freq is exactly the same size as psd
        freq_mask = (freq >= 100) & (freq <= 2000)
        psd_2kHz = psd[freq_mask]

        if len(psd_2kHz) > 0:
            spectral_flatness[k] = gmean(psd_2kHz) / np.mean(psd_2kHz)

    normalized_flatness = spectral_flatness / np.max(spectral_flatness)
    threshold = 0.707
    for k in range(num_frames):
        if normalized_flatness[k] >= threshold:
            noise_cov[k] = np.var(xseg[k, :])
            silent_inds[k] = 1

    R = np.max(noise_cov) # Measurement Noise
    return R, silent_inds

def cmi_tuned_kalman_speech(x, fs):
    p = 15
    y = np.array(x).flatten()

    segment_length = round(0.08 * fs)
    overlap = round(0.01 * fs)
    stride = segment_length - overlap
    totseg = int(np.ceil(len(y) / stride))
    segment = np.zeros((totseg, segment_length))

    # Segment initialization with correct indexing
    start_idx = 0
    for i in range(totseg - 1):
        segment[i, :min(segment_length, len(y) - start_idx)] = y[start_idx:start_idx + segment_length]
        start_idx += stride

    segment[totseg - 1, :len(y) - start_idx] = y[start_idx:]

    H = np.append(np.zeros(p - 1), 1).reshape(1, -1)
    G = H.T
    clean_speech = np.zeros(len(y))

    R, silent_inds = measurement_noise_new(segment, fs)

    X = y[:p].reshape(-1, 1)
    P = np.repeat(R * np.eye(p)[np.newaxis, :, :], segment_length, axis=0)
    Q = 0

    for i in range(totseg):
        A = np.zeros(p)  # Corrected from `p+1` to match expected indexing

        if np.any(np.isnan(segment[i, :])) or np.all(segment[i, :] == 0):
            A = np.zeros(p)
            Q1 = R
        else:
            A = lfilter([1], np.concatenate([-np.array(A[1:]), [1]]), segment[i, :])
            Q1 = np.var(segment[i, :])

        print(f"p: {p}, A shape: {A.shape}, A[1:p-1] shape: {A[1:p-1].shape}")    
        # PHI = np.vstack([np.eye(p - 1), -np.flip(A[1:p])])
        # PHI = np.vstack([np.eye(p-1), -np.flip(A[1:p])])  # Ensuring (15, 15)
        PHI = np.eye(p)
        Q = np.eye(p) * Q1  # Make it match dimensions if needed
        print(f"np.eye(p-1, p).shape: {np.eye(p-1, p).shape}")
        print(f"A[1:p].shape: {A[1:p].shape}")

        
        nq_vals = np.arange(-5, 5)
        J1_vals = np.zeros_like(nq_vals, dtype=float)
        J2_vals = np.zeros_like(nq_vals, dtype=float)

        for q in range(len(nq_vals)):
            Q0 = 10 ** float(nq_vals[q]) * Q1
            H = np.append(np.zeros(p - 2), 1).reshape(1, -1)

            # Ensure correct matrix alignment
            try:
                Ak = H @ (PHI @ P[0] @ PHI.T) @ H.T
                Bk = H @ Q0 @ H.T
            except ValueError as e:
                print(f"Matrix mismatch at segment {i}, iteration {q}: {e}")
                continue

            if np.isnan(Ak) or np.isnan(Bk):
                continue

            J1_vals[q] = R / (Ak + Bk + R)
            J2_vals[q] = Bk / (Ak + Bk)

        nq_nom = np.interp(J1_vals, nq_vals, J2_vals)
        #Q = max(10 ** (nq_nom - 0.7), 1e-6) if nq_nom.size > 0 and not np.isnan(nq_nom) else Q1
        nq_nom = float(nq_nom) if nq_nom.size == 1 else nq_nom.mean()
        Q = max(10 ** (nq_nom - 0.7), 1e-6) if not np.isnan(nq_nom) else Q1
        Q = np.array(Q).reshape(1, 1)  # Ensure Q is at least 1x1

        print(f"PHI shape: {PHI.shape}, X shape: {X.shape}")

        for j in range(segment_length):
            X_ = PHI @ X

            print(f"P[j] shape: {P[j].shape}, PHI shape: {PHI.shape}, G shape: {G.shape}, Q shape: {Q.shape}")

            if P[j].ndim == 0:
                P[j] = np.eye(p) * R  # Ensure it's a matrix
            P_ = (PHI @ P[j] @ PHI.T) + (G @ Q @ G.T)

            if np.any(np.isnan(P_)):
                P_ = np.eye(p) * R

            K = (P_ @ H.T) / max(H @ P_ @ H.T + R, 1e-6)
            P[j + 1] = (np.eye(p) - K @ H) @ P_
            e = segment[i, j] - (H @ X_)

            if np.isnan(e):
                e = 0

            X = X_ + K @ e
            segment[i, j] = X[-1]

    clean_speech[:segment_length] = segment[0, :segment_length]
    start_idx = segment_length
    for i in range(1, totseg - 1):
        clean_speech[start_idx:start_idx + stride] = segment[i, overlap:]
        start_idx += stride

    clean_speech[start_idx:] = segment[-1, :len(y) - start_idx]
    return clean_speech[:len(y)]