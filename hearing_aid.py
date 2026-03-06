import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, stft, istft

# ---------------------------
# Utility helpers
# ---------------------------

def db_to_gain(db):
    """Convert dB (positive = gain, negative = attenuation) to linear gain."""
    return 10 ** (db / 20.0)

def gain_to_db(g):
    """Convert linear gain to dB."""
    return 20.0 * np.log10(np.maximum(g, 1e-12))

# ---------------------------
# Filters: band-splitting
# ---------------------------

def make_band_sos(lowcut, highcut, fs, order=4):
    """Make second-order-sections bandpass (Butterworth) between lowcut and highcut (Hz)."""
    nyq = fs * 0.5
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    if low <= 0:
        # lowpass
        sos = butter(order, high, btype='lowpass', output='sos')
    elif high >= 1:
        # highpass
        sos = butter(order, low, btype='highpass', output='sos')
    else:
        sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos

def split_into_bands(x, fs, bands):
    """Split signal x into list of band signals using SOS filters."""
    band_signals = []
    for (low, high) in bands:
        sos = make_band_sos(low, high, fs)
        # use forward/backward filtering to avoid phase distortion for offline
        y = sosfiltfilt(sos, x)
        band_signals.append(y)
    return band_signals

# ---------------------------
# Simple gain prescription
# ---------------------------

def simple_prescription(audiogram, bands):
    """
    Map audiogram to recommended gain per band (in dB).
    audiogram: dict freq_hz -> hearing threshold (dB HL)
    bands: list of (low, high) tuples.

    This uses a simple heuristic: for each band, find the audiogram threshold
    nearest the band's center frequency and compute gain = alpha * (HL - 20),
    clipped to [0, 40] dB. (HL in dB HL; 0-20 dB is typically normal)
    alpha ~ 0.6 is a gentle prescription factor. Real prescriptive formulas
    (NAL-NL2, DSL) are much more complex.
    """
    alpha = 0.6
    freqs = np.array(sorted(audiogram.keys()))
    thresholds = np.array([audiogram[f] for f in freqs])
    band_gains_db = []
    for (low, high) in bands:
        center = (low + high) / 2.0
        idx = np.argmin(np.abs(freqs - center))
        hl = thresholds[idx]
        gain_db = alpha * max(0.0, hl - 20.0)   # only prescribe for >20 dB HL
        gain_db = float(np.clip(gain_db, 0.0, 40.0))
        band_gains_db.append(gain_db)
    return band_gains_db

# ---------------------------
# Per-band compression
# ---------------------------

class BandCompressor:
    """
    Simple envelope-based compressor applied per-band.
    - threshold_db: level above which compression ratio applies
    - ratio: compression ratio (e.g., 3.0 means 3:1)
    - attack_ms / release_ms: smoothing for gain changes
    """
    def __init__(self, fs, threshold_db= -20.0, ratio=3.0, attack_ms=5.0, release_ms=80.0):
        self.fs = fs
        self.threshold = threshold_db
        self.ratio = ratio
        # smoothing coefficients
        self.attack_coeff = np.exp(-1.0 / (0.001 * attack_ms * fs))
        self.release_coeff = np.exp(-1.0 / (0.001 * release_ms * fs))
        self.env = 0.0

    def process(self, x, makeup_db=0.0):
        """
        x: single-channel time-domain band signal (numpy array)
        returns compressed band signal
        """
        out = np.zeros_like(x)
        # simple envelope detector on absolute value
        for n, s in enumerate(x):
            rect = abs(s)
            if rect > self.env:
                self.env = self.attack_coeff * (self.env - rect) + rect
            else:
                self.env = self.release_coeff * (self.env - rect) + rect
            # convert env to dB FS (approx)
            env_db = 20.0 * np.log10(max(self.env, 1e-12))
            if env_db > self.threshold:
                # amount above threshold
                above = env_db - self.threshold
                gain_reduction_db = above - (above / self.ratio)
            else:
                gain_reduction_db = 0.0
            gain_lin = db_to_gain(-gain_reduction_db + makeup_db)
            out[n] = s * gain_lin
        return out

# ---------------------------
# Spectral subtraction (simple noise reduction)
# ---------------------------

def spectral_subtraction_denoise(x, fs, n_fft=1024, hop=512, noise_frames=6, over_sub=1.0, floor_db=-20.0):
    """
    Very simple spectral subtraction:
      - estimate noise magnitude from first `noise_frames` frames (assumes noise-only at start)
      - subtract scaled noise spectrum from each frame
    This is only for demo; practical systems use improved Wiener, MMSE, or DNN methods.
    """
    f, t, Zxx = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    # estimate noise magnitude spectrum as average of first frames
    noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    # subtract
    mag_denoised = np.maximum(mag - over_sub * noise_mag, 10**(floor_db/20.0))
    # reconstruct
    Zxx_denoised = mag_denoised * np.exp(1j * phase)
    _, xrec = istft(Zxx_denoised, fs=fs, nperseg=n_fft, noverlap=n_fft - hop, input_onesided=True, boundary=True)
    # trim/pad to original length
    xrec = xrec[:len(x)]
    return xrec

# ---------------------------
# End-to-end processing
# ---------------------------

def process_wav(input_path, output_path, audiogram):
    x, fs = sf.read(input_path, dtype='float32')
    if x.ndim > 1:
        # mix to mono for this prototype
        x = np.mean(x, axis=1)
    print(f"Read '{input_path}': {len(x)} samples, {fs} Hz")

    # define bands (approx octave bands covering 125 Hz to 8000 Hz)
    bands = [
        (50, 250),   # low
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
    ]

    # 1) Noise reduction (frame spectral subtraction)
    print("Applying spectral-subtraction noise reduction...")
    x_nr = spectral_subtraction_denoise(x, fs, n_fft=2048, hop=512, noise_frames=6, over_sub=1.0)

    # 2) Split into bands
    print("Splitting into bands...")
    band_signals = split_into_bands(x_nr, fs, bands)

    # 3) Compute per-band prescription gain (dB)
    band_gains_db = simple_prescription(audiogram, bands)
    print("Band gain targets (dB):", band_gains_db)

    # 4) Per-band processing: apply gain + compressor
    processed_bands = []
    for i, band in enumerate(bands):
        y = band_signals[i]
        # apply prescribed gain (makeup) before compression (one design choice)
        makeup_db = band_gains_db[i]
        # compressor settings can vary by band (low freq might use different ratio)
        comp = BandCompressor(fs, threshold_db=-30.0 + (i * 2.0), ratio=2.5 + i*0.2, attack_ms=5.0, release_ms=60.0)
        y_comp = comp.process(y, makeup_db=makeup_db)
        processed_bands.append(y_comp)

    # 5) Recombine bands (sum) and limit
    print("Recombining bands...")
    y_sum = np.sum(np.stack(processed_bands, axis=0), axis=0)
    # Normalize to avoid clipping (simple limiter)
    peak = np.max(np.abs(y_sum))
    if peak > 0.98:
        y_sum = y_sum * (0.98 / peak)

    # 6) Write output
    sf.write(output_path, y_sum, fs)
    print(f"Wrote processed file to '{output_path}'")

# ---------------------------
# Example audiogram (Hz -> dB HL)
# ---------------------------
# Typical audiogram frequencies commonly tested: 250, 500, 1000, 2000, 4000, 8000
# Values are hearing thresholds in dB HL (0 = normal hearing)
EXAMPLE_AUDIOGRAM = {
    250: 25.0,
    500: 30.0,
    1000: 40.0,
    2000: 45.0,
    4000: 50.0,
    8000: 55.0,
}

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python hearing_aid.py input.wav output.wav")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2]
    process_wav(inp, outp, EXAMPLE_AUDIOGRAM)