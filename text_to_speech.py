import os
import requests
import math
from dotenv import load_dotenv

# Ensure the temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

class TextToSpeech:
    def __init__(self, api_key):
        """
        Initialize ElevenLabs Text-to-Speech service
        
        Args:
            api_key (str): ElevenLabs API key
        """
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

    def split_text(self, text, max_chars=2000):
        """
        Split long text into chunks
        
        Args:
            text (str): Input text
            max_chars (int): Maximum characters per chunk
        
        Returns:
            list: List of text chunks
        """
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
        return chunks

    def text_to_speech(self, text, voice_id='21m00Tcm4TlvDq8ikWAM', language='en'):
        """
        Convert text to speech using ElevenLabs API
        
        Args:
            text (str): Text to convert
            voice_id (str): Voice ID to use
            language (str): Language of the text
        
        Returns:
            str: Path to generated audio file
        """
        try:
            # Split text into chunks
            text_chunks = self.split_text(text)
            
            # Prepare output audio path
            output_audio = os.path.join(TEMP_DIR, f'translated_speech_{language}.mp3')
            
            # Process chunks
            with open(output_audio, 'wb') as audio_file:
                for chunk in text_chunks:
                    # Prepare API payload
                    payload = {
                        "text": chunk,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.5
                        }
                    }
                    
                    # Make API request
                    response = requests.post(
                        f"{self.base_url}/text-to-speech/{voice_id}",
                        json=payload,
                        headers=self.headers
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        # Append chunk to audio file
                        audio_file.write(response.content)
                    else:
                        print(f"Error processing chunk: {response.text}")
                        return None
            
            print(f"Text-to-Speech conversion successful.")
            print(f"Audio saved to: {output_audio}")
            return output_audio
        
        except Exception as e:
            print(f"Text-to-Speech conversion error: {e}")
            return None

def main():
    
    load_dotenv()
    API_KEY = os.getenv('ELEVENLABS_API_KEY')
    if not API_KEY:
        print("Error: ELEVENLABS_API_KEY not found in .env file.")
        return
    
    # Read the translated text
    translation_path = os.path.join(TEMP_DIR, 'translated_text.txt')
    
    try:
        with open(translation_path, 'r', encoding='utf-8') as f:
            translated_text = f.read()
    except FileNotFoundError:
        print("Translated text file not found.")
        return
    
    # Initialize TTS
    tts = TextToSpeech(API_KEY)
    
    # Voice options (some examples)
    voices = {
        'en': '21m00Tcm4TlvDq8ikWAM',  # Male English
        'fr': 'fr-FR-Standard-E',      # French
        'es': 'es-ES-Standard-A',      # Spanish
        'de': 'de-DE-Standard-A',      # German
    }
    
    # Prompt for language of speech generation
    print("Enter the language code for speech generation (e.g., 'en', 'fr', 'es'):")
    language = input().strip().lower()
    
    # Get appropriate voice ID
    voice_id = voices.get(language, '21m00Tcm4TlvDq8ikWAM')
    
    # Convert text to speech
    speech_file = tts.text_to_speech(translated_text, voice_id, language)
    
    if speech_file:
        print(f"Speech audio generated: {speech_file}")
    else:
        print("Failed to generate speech audio.")

if __name__ == "__main__":
    main()