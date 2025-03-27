import os
from googletrans import Translator, LANGUAGES

# Ensure the temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def list_supported_languages():
    """
    List all supported languages with their codes.
    
    Returns:
        dict: Dictionary of language codes and names
    """
    return LANGUAGES

def translate_text(text, target_language='en'):
    """
    Translate text to the target language.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (default is English)
    
    Returns:
        dict: Translation result with details
    """
    try:
        # Initialize translator
        translator = Translator()
        
        # Detect source language
        detection = translator.detect(text)
        
        # Translate text
        translation = translator.translate(text, dest=target_language)
        
        # Prepare result dictionary
        result = {
            'original_text': text,
            'original_language': detection.lang,
            'translated_text': translation.text,
            'target_language': target_language
        }
        
        # Save translated text
        translation_path = os.path.join(TEMP_DIR, 'translated_text.txt')
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(translation.text)
        
        # Print details
        print(f"Translation successful.")
        print(f"Original Language: {detection.lang}")
        print(f"Target Language: {target_language}")
        print(f"Translation saved to: {translation_path}")
        
        return result
    
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    # Print supported languages for reference
    print("Supported Languages:")
    for code, name in LANGUAGES.items():
        print(f"{code}: {name}")
    
    # Example usage
    # Read the transcribed text
    transcription_path = os.path.join(TEMP_DIR, 'full_transcription.txt')
    
    with open(transcription_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Prompt for target language
    print("\nEnter the target language code (e.g., 'fr' for French, 'es' for Spanish):")
    target_language = input().strip().lower()
    
    # Translate
    translation_result = translate_text(text, target_language)
    
    if translation_result:
        print("\nOriginal Text:", translation_result['original_text'])
        print("Translated Text:", translation_result['translated_text'])
    else:
        print("Translation failed.")

if __name__ == "__main__":
    main()