import os
from faster_whisper import WhisperModel

# Ensure the temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def speech_to_text(audio_path, model_size='base', device='auto'):
    """
    Convert audio to text using Faster Whisper.
    
    Args:
        audio_path (str): Path to the input audio file
        model_size (str): Size of the Whisper model to use
        device (str): Compute device ('auto', 'cuda', 'cpu')
    
    Returns:
        dict: Transcription details including language, text
    """
    try:
        # Initialize the Whisper model
        # Model sizes: 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'
        model = WhisperModel(
            model_size, 
            device=device,  # 'auto' will use CUDA if available, else CPU
            compute_type='float16' if device == 'cuda' else 'float32'
        )
        
        # Transcribe the audio
        segments, info = model.transcribe(
            audio_path, 
            beam_size=5,  # Improves accuracy
            language=None,  # Auto-detect language
            condition_on_previous_text=False
        )
        
        # Combine transcribed segments
        full_text = ' '.join(segment.text for segment in segments)
        
        # Save full transcription
        transcription_path = os.path.join(TEMP_DIR, 'full_transcription.txt')
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        # Prepare result dictionary
        result = {
            'text': full_text,
            'language': info.language,
            'language_probability': info.language_probability
        }
        
        # Print details
        print(f"Transcription successful.")
        print(f"Detected Language: {info.language}")
        print(f"Language Probability: {info.language_probability:.2%}")
        print(f"Transcription saved to: {transcription_path}")
        
        return result
    
    except Exception as e:
        print(f"Error in speech to text conversion: {e}")
        return None

def main():
    # Example usage
    audio_path = os.path.join(TEMP_DIR, [f for f in os.listdir(TEMP_DIR) if f.endswith('_audio.wav')][0])
    transcription_result = speech_to_text(audio_path)
    
    if transcription_result:
        print(f"Detected Language: {transcription_result['language']}")
        print(f"Transcribed Text: {transcription_result['text']}")
    else:
        print("Speech to text conversion failed.")

if __name__ == "__main__":
    main()