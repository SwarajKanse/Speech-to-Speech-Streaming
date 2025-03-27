import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import google.generativeai as genai
    from google.generativeai.types import generation_types
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    sys.exit(1)

try:
    from googletrans import Translator as GoogleTranslator
except ImportError:
    print("Please install googletrans: pip install googletrans==3.1.0a0")
    sys.exit(1)

class VideoTranslator:
    def __init__(self, 
                 gemini_api_key=None, 
                 gemini_model='gemini-1.5-flash'):
        """
        Initialize the video translator with Gemini API key and model.
        
        Args:
            gemini_api_key (str, optional): API key for Google Gemini. 
                                            If not provided, will use environment variable.
            gemini_model (str, optional): Gemini model to use for translation.
        """
        # Try to get API key from environment or parameter
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            raise ValueError("No Gemini API key found. Please set GEMINI_API_KEY in .env file.")
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(gemini_model)
        
        # Initialize Google Translate as fallback
        self.google_translator = GoogleTranslator()

    def translate_text(self, text, target_language):
        """
        Translate text using Gemini.
        
        Args:
            text (str): Text to translate
            target_language (str): Target language code (e.g., 'fr' for French)
        
        Returns:
            str: Translated text
        """
        if not text:
            return ""

        try:
            # Translate using Gemini
            prompt = f"Translate the following text to {target_language}: '{text}'"
            response = self.gemini_model.generate_content(prompt)
            
            # Check if generation was successful
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # If Gemini fails, fall back to Google Translate
            return self.google_translator.translate(text, dest=target_language).text
        
        except Exception as e:
            print(f"Translation error: {e}")
            # Fallback to Google Translate if Gemini fails
            return self.google_translator.translate(text, dest=target_language).text

    def translate_transcription(self, transcription_path, target_language, script_dir=None):
        """
        Translate entire transcription file.
        
        Args:
            transcription_path (str): Path to transcription text file
            target_language (str): Target language code
            script_dir (str, optional): Directory to save translated file
        
        Returns:
            str: Path to translated transcription file
        """
        # If no directory provided, use current script's directory
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        # Read transcription
        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_text = f.read()
        except Exception as e:
            print(f"Error reading transcription: {e}")
            return None

        # Translate transcription
        translated_text = self.translate_text(transcription_text, target_language)

        # Generate translated filename in the script's directory
        original_filename = os.path.basename(transcription_path)
        translated_filename = original_filename.replace('.txt', f'_{target_language}.txt')
        translated_path = os.path.join(script_dir, translated_filename)

        # Write translated transcription
        try:
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            print(f"Translated transcription saved to {translated_path}")
            return translated_path
        except Exception as e:
            print(f"Error saving translated transcription: {e}")
            return None

# Example usage
def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration
    transcription_path = os.path.join(script_dir, "transcription.txt")
    target_language = "hi"  # Hindi as an example

    try:
        # Initialize translator with Gemini 1.5 Flash
        translator = VideoTranslator()

        # Translate transcription, saving in the script's directory
        translated_path = translator.translate_transcription(
            transcription_path, 
            target_language,
            script_dir
        )

        if translated_path:
            print(f"Translation complete. Output: {translated_path}")
        else:
            print("Translation failed.")

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    main()