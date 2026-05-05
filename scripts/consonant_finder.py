import numpy as np
import soundfile as sf
from scipy.signal import correlate



def estimate_s_timings_seconds(
    audiodata,
    samplerate,
    frame_ms=20,
    hop_ms=5,
    min_f0=60,
    max_f0=350,
    quiet_rms_threshold=0.01,
    voiced_periodicity_threshold=0.3,
    hf_cutoff=4000,
    hf_energy_ratio_threshold=0.25,
):
    """
    Detect /s/ intervals using:
      1) low periodicity (unvoiced)
      2) high-frequency energy above hf_cutoff

    Returns
    -------
    intervals : list of (start_time, end_time)
    """

    frame = int(samplerate * frame_ms / 1000)
    hop = int(samplerate * hop_ms / 1000)
    min_lag = int(samplerate / max_f0)
    max_lag = int(samplerate / min_f0)

    intervals = []
    in_s = False
    start_time = None

    for start in range(0, len(audiodata) - frame + 1, hop):
        w = audiodata[start:start + frame]
        w = w - np.mean(w)

        rms = np.sqrt(np.mean(w**2))
        if rms < quiet_rms_threshold:
            if in_s:
                intervals.append((start_time, start / samplerate))
                in_s = False
                start_time = None
            continue

        # Autocorrelation periodicity
        ac = correlate(w, w, mode="full")[frame - 1:]
        ac /= (ac[0] + 1e-12)

        ml = max(min_lag, 1)
        Ml = min(max_lag, len(ac) - 1)

        periodicity = 0.0
        if ml <= Ml:
            periodicity = float(np.max(ac[ml:Ml + 1]))

        # High-frequency energy ratio above 4 kHz
        spec = np.abs(np.fft.rfft(w)) ** 2
        freqs = np.fft.rfftfreq(len(w), d=1.0 / samplerate)

        total_energy = np.sum(spec) + 1e-12
        hf_energy = np.sum(spec[freqs >= hf_cutoff])
        hf_ratio = hf_energy / total_energy

        # /s/ criterion: unvoiced + high-frequency noise-like energy
        is_s_like = (
            periodicity < voiced_periodicity_threshold
            and hf_ratio >= hf_energy_ratio_threshold
        )

        if not in_s and is_s_like:
            in_s = True
            start_time = start / samplerate

        elif in_s and not is_s_like:
            end_time = start / samplerate
            intervals.append((start_time, end_time))
            in_s = False
            start_time = None

    # close final interval if file ends while still in /s/
    if in_s and start_time is not None:
        intervals.append((start_time, len(audiodata) / samplerate))

    return intervals

if __name__ == "__main__":
    audio_file = "audiofiles/Say.wav"
    x, sr = sf.read(audio_file)
    if x.ndim > 1:
        x = x.mean(axis=1)

    t = estimate_s_timings_seconds(x, sr)
    print(t)