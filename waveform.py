import matplotlib.pyplot as plt
import numpy as np
import wave
from consonant_finder import estimate_s_timings_seconds

# Open WAV file
def waveform(spf):
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int16)
    signal = signal / np.max(np.abs(signal))

    fs= spf.getframerate()
    time= np.linspace(0, len(signal)/fs, num= len(signal))

    # Create plot
    plt.plot(time, signal)
    plt.title(file)
    #onset= estimate_s_timings_seconds(file)
    #plt.axvline(x= onset, color= "red", label= f"voice onset time: {onset}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)")
    plt.legend()
    plt.tight_layout()

if __name__ == "__main__":
    file= input("input .wav file: ")
    spf = wave.open(file, 'r')
    plt.subplot(2, 1, 1)
    waveform(spf)

    file= input("input .wav file: ")
    spf = wave.open(file, 'r')
    plt.subplot(2, 1, 2)
    waveform(spf)
    plt.tight_layout()
    plt.show()