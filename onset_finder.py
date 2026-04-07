import numpy as np
import soundfile as sf
from scipy.signal import correlate

def estimate_voicing_onset_seconds(path, frame_ms=20, hop_ms=5, min_f0=60, max_f0=350):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)

    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)

    best_time = None

    for start in range(0, len(x) - frame, hop):
        w = x[start:start + frame]
        w = w - np.mean(w)

        # skip very quiet frames
        if np.sqrt(np.mean(w**2)) < 0.01:
            continue

        ac = correlate(w, w, mode="full")[frame - 1:]
        ac /= (ac[0] + 1e-12)

        periodicity = np.max(ac[min_lag:max_lag + 1])

        # threshold may need tuning for your recording
        if periodicity > 0.3:
            best_time = start / sr
            break

    return best_time

if __name__ == "__main__":
    t = estimate_voicing_onset_seconds("Say.wav")
    print(t)