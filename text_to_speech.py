import os
import sys
import traceback
from pathlib import Path
import math

# Import the VideoTranslator from the previous script
from translation import VideoTranslator

try:
    import torch
    import torchaudio
    from tortoise.api import TextToSpeech
except ImportError:
    print("Please install required packages:")
    print("pip install tortoise-tts torch torchaudio")
    sys.exit(1)

class TranslatedTextToSpeech:
    def __init__(self, output_dir=None, preset="fast", max_segment_length=200):
        """
        Initialize Text-to-Speech with Tortoise TTS and text segmentation.
        
        Args:
            output_dir (str, optional): Directory to save audio files
            preset (str, optional): TTS generation preset
            max_segment_length (int, optional): Max characters per segment
        """
        # Ensure CUDA is available if possible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize TTS model
        try:
            self.tts = TextToSpeech()
            self.preset = preset
            self.max_segment_length = max_segment_length
        except Exception as e:
            print(f"Error initializing TTS model: {e}")
            raise

        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _split_text(self, text):
        """
        Split text into segments of roughly equal length.
        
        Args:
            text (str): Input text to split
        
        Returns:
            list: List of text segments
        """
        # Split text into sentences if possible
        sentences = []
        current_segment = ""
        
        # Prefer splitting by punctuation marks first
        for sentence in text.split('.'):
            sentence = sentence.strip() + '.'
            
            # If adding this sentence would make the segment too long, start a new segment
            if len(current_segment) + len(sentence) > self.max_segment_length:
                sentences.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += ' ' + sentence
        
        # Add the last segment if not empty
        if current_segment.strip():
            sentences.append(current_segment.strip())
        
        return sentences

    def _clone_voice_samples(self, voice_samples_path):
        """
        Clone voice samples for Tortoise TTS with multiple variations.
        
        Args:
            voice_samples_path (str): Path to input audio file
        
        Returns:
            list: List of cloned voice sample tensors
        """
        try:
            # Load original voice samples
            voice_samples, sr = torchaudio.load(voice_samples_path)
            
            # Ensure audio is mono
            if voice_samples.ndim > 1:
                voice_samples = voice_samples.mean(dim=0, keepdim=True)
            
            # Resample if necessary (Tortoise expects 24kHz)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                voice_samples = resampler(voice_samples)
            
            # Ensure 2D tensor with shape [1, num_samples]
            voice_samples = voice_samples.squeeze()
            
            # Ensure minimum length for voice samples
            min_length = 24000 * 5  # 5 seconds minimum
            max_length = 24000 * 60  # Max 1 minute
            
            if voice_samples.numel() < min_length:
                # Repeat the sample to meet minimum length
                repeats = (min_length // voice_samples.numel()) + 1
                voice_samples = voice_samples.repeat(repeats)
            
            # Truncate to a reasonable length
            voice_samples = voice_samples[:max_length]
            
            # Create multiple variations of the voice sample
            voice_variations = []
            
            # Original sample
            voice_variations.append(voice_samples.unsqueeze(0))
            
            # Add some variations by slightly modifying the original sample
            for _ in range(2):
                # Create a slightly modified version by adding low-level noise
                noisy_sample = voice_samples + torch.randn_like(voice_samples) * 0.01
                voice_variations.append(noisy_sample.unsqueeze(0))
            
            return voice_variations
        
        except Exception as e:
            print(f"Error cloning voice samples: {e}")
            # Create default noise tensors if voice sample loading fails
            default_samples = [torch.randn(1, 24000 * 5) for _ in range(3)]
            return default_samples

    def text_to_speech(self, 
                       text_file_path, 
                       voice_samples_path="D:\\Infosys Springboard\\Basic Project\\input_audio.wav"):
        """
        Convert translated text to speech using Tortoise TTS with voice cloning.
        
        Args:
            text_file_path (str): Path to translated text file
            voice_samples_path (str, optional): Path to voice samples for cloning
        
        Returns:
            str: Path to generated audio file
        """
        # Read translated text
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None

        # Generate output filename
        text_filename = os.path.basename(text_file_path)
        audio_filename = text_filename.replace('.txt', '.wav')
        output_path = self.output_dir / audio_filename

        try:
            # Clone voice samples with multiple variations
            voice_samples_list = self._clone_voice_samples(voice_samples_path)
            
            # Split text into segments
            text_segments = self._split_text(text)
            
            # Prepare list to concatenate audio segments
            audio_segments = []

            # Generate speech for each segment using different voice sample variations
            for i, segment in enumerate(text_segments):
                print(f"Generating segment {i+1}/{len(text_segments)}: {segment[:50]}...")
                
                # Cycle through voice sample variations
                voice_samples = voice_samples_list[i % len(voice_samples_list)]
                
                # Generate with voice cloning
                gen = self.tts.tts_with_preset(
                    segment, 
                    voice_samples=voice_samples, 
                    preset=self.preset
                )
                
                audio_segments.append(gen)

            # Concatenate all audio segments
            final_audio = torch.cat(audio_segments, dim=-1)

            # Save the generated audio
            torchaudio.save(str(output_path), final_audio.squeeze(0), 24000)

            print(f"Audio generated: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"TTS generation error: {e}")
            print(f"Error details: {traceback.format_exc()}")
            return None

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the most recently created translated file
    translated_files = [f for f in os.listdir(script_dir) if f.endswith('_fr.txt')]
    
    if not translated_files:
        print("No translated files found. Run translation first.")
        return

    # Use the most recently created translated file
    translated_path = os.path.join(script_dir, max(translated_files, key=lambda f: os.path.getctime(os.path.join(script_dir, f))))
    
    # Path to input audio for voice cloning
    voice_samples_path = "D:\\Infosys Springboard\\Basic Project\\input_audio.wav"
    
    try:
        # Convert translated text to speech
        tts_generator = TranslatedTextToSpeech(
            output_dir=script_dir, 
            max_segment_length=250  # Adjust this value as needed
        )
        
        audio_path = tts_generator.text_to_speech(
            translated_path,
            voice_samples_path=voice_samples_path
        )

        if audio_path:
            print(f"Text-to-Speech generation complete. Audio: {audio_path}")
        else:
            print("Text-to-Speech generation failed.")

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    main()