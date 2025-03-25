import whisper

def transcribe_audio(audio_path):
    model = whisper.load_model("medium")  # You can use "tiny", "base", "small", "medium", or "large"
    result = model.transcribe(audio_path)
    return result["text"]

# Example usage:
audio_path = "extracted_audio.wav"  # Change this to your actual file path
transcription = transcribe_audio(audio_path)

# Optionally, save to a text file
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(transcription)