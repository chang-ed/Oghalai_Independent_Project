import numpy as np
import soundfile as sf
from onset_finder import estimate_voicing_onset_seconds

def consonant_changer(
    wav_in: str,
    wav_out: str,
    voicing_onset_s: float,
    pre_noise_ms: float = 30.0,
    voiced_template_ms: float = 80.0,
    fade_ms: float = 5.0,
):
    """
    Idea: modify fricative consonant "envelopes"

    Parameters
    ----------
    wav_in : input WAV path
    wav_out : output WAV path
    voicing_onset_s : onset of voicing in seconds
    pre_noise_ms : amount of fricative noise to sample before onset
    voiced_template_ms : amount of voiced speech to use as the template
    fade_ms : crossfade length to avoid clicks
    """
    x, sr = sf.read(wav_in)
    if x.ndim > 1:
        x = np.mean(x, axis=1)  # convert to mono

    onset = int(round(voicing_onset_s * sr))
    shift = int(round(140.0 * sr / 1000.0)) #delay offset as an example
    pre_noise = int(round(pre_noise_ms * sr / 1000.0))
    voiced_len = int(round(voiced_template_ms * sr / 1000.0))
    fade = int(round(fade_ms * sr / 1000.0))

    if onset <= 0 or onset >= len(x):
        raise ValueError("voicing_onset_s must fall inside the file")

    # Build source snippets
    noise_start = max(0, onset - pre_noise)
    noise_src = x[noise_start:onset]
    if len(noise_src) < 10:
        raise ValueError("Not enough pre-onset material to estimate fricative noise")

    voiced_end = min(len(x), onset + voiced_len)
    voiced_src = x[onset:voiced_end]
    if len(voiced_src) < 10:
        raise ValueError("Not enough voiced material after onset to use as a template")

    y = x.copy()

    # Delay voicing: replace the interval [onset, onset+shift] with repeated fricative noise.
    silence = np.zeros(shift, dtype=x.dtype)
    y = np.concatenate([
            x[:onset],
            silence,
            x[onset:]
        ])

    sf.write(wav_out, y, sr)


if __name__ == "__main__":
    # Example:
    # Move voicing onset 140 ms later
    input_file= input("Enter .wav to be changed:\n")
    VOT_OG= estimate_voicing_onset_seconds(input_file)
    consonant_changer(
        wav_in=input_file,
        wav_out="140_ms_VOT.wav",
        voicing_onset_s=VOT_OG,
    )