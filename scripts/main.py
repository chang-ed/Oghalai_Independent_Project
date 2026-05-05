import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from consonant_finder import estimate_s_timings_seconds
from detector import find_words

def main():
    audio_file= "audiofiles/harvard.wav"
    transcript_plus_timings= find_words(audio_file= audio_file)

    x, sr = sf.read(audio_file)
    if x.ndim > 1:
        x = x.mean(axis=1)


    for w in transcript_plus_timings["word_segments"]:
        word = w['word']
        if 's' in word.lower(): #need to implement all fricative consonants 
            #/f, v, θ, ð, s, z, ʃ, ʒ, h/.
    #problem: s can make both /s/ and /sh/ sounds. Future: implement a feature that can distinguish based on transcript
            start_sec = float(w["start"])
            end_sec = float(w["end"])

            start_sample = max(0, int(start_sec * sr))
            end_sample = min(len(x), int(end_sec * sr))

            clip = x[start_sample:end_sample]

            consonant_intervals= estimate_s_timings_seconds(clip, sr, voiced_periodicity_threshold= 0.5)



            #add code modifying original sound file here!!




            if word == "restores":
                print(f"\nThere are /s/ at", sep= " ")
                for i,_ in enumerate(consonant_intervals):
                    print(f"{consonant_intervals[i][0]}")
                print(f"\nRestores is {end_sec - start_sec} sec long")

                plt.subplot(2, 1, 1)
                plt.specgram(clip, Fs= sr, cmap='viridis')
                plt.title('Spectrogram of Restores')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.colorbar(label='Intensity [dB]')
                for start, end in consonant_intervals:
                    plt.axvspan(start, end, color='red', alpha=0.3)

                time= np.arange(len(clip)) / sr
                plt.subplot(2, 1, 2)
                plt.plot(time, clip)
                plt.title("Waveform")
                plt.xlabel("Time [sec]")
                plt.ylabel("Amplitude")
                for start, end in consonant_intervals:
                    plt.axvspan(start, end, color='red', alpha=0.3)

                plt.tight_layout()
                plt.show()

                plt.specgram(clip, Fs= sr, cmap='viridis')
                plt.title('Spectrogram of Restores')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.colorbar(label='Intensity [dB]')
                plt.show()





if __name__ == "__main__":
    main()