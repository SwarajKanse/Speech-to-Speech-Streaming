import os
import sys
import traceback

# Attempt to import libraries
try:
    import torch
    import faster_whisper
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'faster-whisper', 'torch'])
    import faster_whisper
    import torch

class AdvancedTranscriber:
    def __init__(self, model_size='medium', device=None):
        """
        Initialize transcription with optimized settings.
        
        Args:
            model_size (str): Whisper model size 
            device (str, optional): Compute device
        """
        # Automatic device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device}")
        
        # Model selection with performance optimization
        self.model = faster_whisper.WhisperModel(
            model_size, 
            device=device, 
            compute_type='float16' if device == 'cuda' else 'float32',
            cpu_threads=os.cpu_count() // 2  # Utilize half the CPU cores
        )

    def transcribe(self, 
                   audio_path, 
                   language=None, 
                   task='transcribe', 
                   beam_size=5):
        """
        Transcribe audio with advanced configurations.
        
        Args:
            audio_path (str): Path to audio file
            language (str, optional): Language code (auto-detect if None)
            task (str): 'transcribe' or 'translate'
            beam_size (int): Beam size for decoding (higher = more accurate, slower)
        
        Returns:
            str: Transcribed text
        """
        try:
            # Transcription options
            options = {
                'beam_size': beam_size,
                'best_of': 5,  # Number of alternative transcriptions to consider
                'patience': 1.0,  # Patience factor for generation
                'length_penalty': 1.0,  # Penalty for longer/shorter translations
            }
            
            # Add language if specified
            if language:
                options['language'] = language

            # Perform transcription
            segments, info = self.model.transcribe(
                audio_path, 
                **options
            )

            # Combine segments
            transcription = ' '.join(segment.text for segment in segments)
            
            # Print detected language if not specified
            if not language:
                print(f"Detected Language: {info.language} (Probability: {info.language_probability:.2f})")
            
            return transcription

        except Exception as e:
            print(f"Transcription Error: {e}")
            traceback.print_exc()
            return ""

    def save_transcription(self, transcription, script_dir=None, output_filename='transcription.txt'):
        """
        Save transcription to a text file in the script's directory.
        
        Args:
            transcription (str): Transcribed text
            script_dir (str, optional): Directory to save the file
            output_filename (str): Name of the output file
        """
        try:
            # If no directory provided, use current script's directory
            if script_dir is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create full path for output file
            output_path = os.path.join(script_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"Transcription saved to: {output_path}")
        except Exception as e:
            print(f"Error saving transcription: {e}")

def find_audio_file(script_dir):
    """
    Find the first audio file in the script's directory.
    
    Args:
        script_dir (str): Directory of the script
    
    Returns:
        str: Path to the first found audio file or None
    """
    # List of common audio file extensions
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    
    # Search for audio files
    for filename in os.listdir(script_dir):
        if any(filename.lower().endswith(ext) for ext in audio_extensions):
            return os.path.join(script_dir, filename)
    
    return None

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find audio file in the same directory
    audio_path = find_audio_file(script_dir)
    
    # Validate audio file
    if not audio_path:
        print("Error: No audio file found in the script's directory.")
        print("Supported formats: .wav, .mp3, .m4a, .flac, .ogg")
        return

    print(f"Found audio file: {audio_path}")

    # Initialize transcriber
    transcriber = AdvancedTranscriber(model_size='medium')

    # Transcribe audio (auto-detect language)
    transcription = transcriber.transcribe(audio_path)

    # Save transcription in the same directory as the script
    if transcription:
        transcriber.save_transcription(transcription, script_dir)
    else:
        print("No transcription generated.")

if __name__ == "__main__":
    main()