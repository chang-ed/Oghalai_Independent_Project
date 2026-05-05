import whisperx

def find_words(audio_file: str):

    device = "cpu"
    batch_size = 1
    compute_type = "int8"

    model = whisperx.load_model("small", device, compute_type=compute_type) #don't have space on laptop for large-v2


    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size) # before alignment

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    return result

if __name__ == "__main__":
    audio_file= "audiofiles/harvard.wav"
    transcript= find_words(audio_file= audio_file)
    print("\n")
    print(transcript)
    print("\n")
    print(transcript["segments"])