import os
import sys
import re
import shutil
from dotenv import load_dotenv

# Extensive import for multiple download methods
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from pytube import YouTube
except ImportError:
    YouTube = None

# Import other script modules
from audio_extraction import extract_audio
from speech_to_text import speech_to_text
from translation import translate_text
from text_to_speech import TextToSpeech
from audio_insertion import synchronize_and_generate_video

# Ensure the temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def is_youtube_url(url):
    """
    Comprehensive check for YouTube URLs.
    
    Args:
        url (str): URL to check
    
    Returns:
        bool: True if it's a YouTube URL, False otherwise
    """
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return re.match(youtube_regex, url) is not None

def download_with_yt_dlp(url):
    """
    Download YouTube video using yt-dlp.
    
    Args:
        url (str): YouTube video URL
    
    Returns:
        str: Path to downloaded video or None
    """
    if yt_dlp is None:
        print("yt-dlp is not installed. Skipping yt-dlp method.")
        return None

    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(TEMP_DIR, 'youtube_video_%(id)s.%(ext)s'),
            'nooverwrites': True,
            'no_color': True,
            'ignoreerrors': False,
            'no_warnings': True,
            'quiet': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_filename = ydl.prepare_filename(info_dict)
            
            print(f"yt-dlp Download Successful:")
            print(f"Video Title: {info_dict.get('title', 'Unknown')}")
            print(f"Video Duration: {info_dict.get('duration', 'Unknown')} seconds")
            print(f"Downloaded to: {video_filename}")
            
            return video_filename
    
    except Exception as e:
        print(f"yt-dlp download error: {e}")
        return None

def download_with_pytube(url):
    """
    Download YouTube video using pytube.
    
    Args:
        url (str): YouTube video URL
    
    Returns:
        str: Path to downloaded video or None
    """
    if YouTube is None:
        print("pytube is not installed. Skipping pytube method.")
        return None

    try:
        # Create YouTube object
        yt = YouTube(url)
        
        # Get the highest resolution mp4 video stream
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').last()
        
        # Prepare output path
        output_filename = f'youtube_video_{yt.video_id}.mp4'
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Download the video
        video_stream.download(output_path=TEMP_DIR, filename=output_filename)
        
        print(f"pytube Download Successful:")
        print(f"Video Title: {yt.title}")
        print(f"Video Duration: {yt.length} seconds")
        print(f"Downloaded to: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"pytube download error: {e}")
        return None

def download_youtube_video(url):
    """
    Attempt to download YouTube video using multiple methods.
    
    Args:
        url (str): YouTube video URL
    
    Returns:
        str: Path to downloaded video or None
    """
    # List of download methods to try
    download_methods = [
        download_with_yt_dlp,
        download_with_pytube
    ]
    
    # Try each download method
    for method in download_methods:
        try:
            video_path = method(url)
            if video_path and os.path.exists(video_path):
                return video_path
        except Exception as e:
            print(f"Download method failed: {method.__name__}")
            print(f"Error: {e}")
    
    print("All download methods failed.")
    return None

def clean_temp_directory(input_video_path=None, keep_files=None):
    """
    Clean the temp directory, keeping specified files and input video.
    
    Args:
        input_video_path (str): Path to the input video to keep
        keep_files (list): List of additional filenames to keep
    """
    keep_files = keep_files or []
    
    # Add input video filename to keep_files if provided
    if input_video_path:
        keep_files.append(os.path.basename(input_video_path))
    
    # Add an entry to ensure final translated video is deleted
    keep_files = [f for f in keep_files if f != 'final_translated_video.mp4']
    
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        
        # Skip files that should be kept
        if filename in keep_files:
            continue
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def translate_video(input_video_path, target_language, elevenlabs_api_key):
    """
    Comprehensive workflow for video language translation.
    
    Args:
        input_video_path (str): Path to the input video
        target_language (str): Target language code for translation
        elevenlabs_api_key (str): API key for ElevenLabs text-to-speech
    
    Returns:
        str: Path to the final translated video
    """
    try:
        # Clean temp directory, keeping the input video
        clean_temp_directory(input_video_path)
        
        # Step 1: Extract Audio
        extracted_audio = extract_audio(input_video_path)
        if not extracted_audio:
            raise Exception("Audio extraction failed")
        
        # Step 2: Speech to Text
        transcription = speech_to_text(extracted_audio)
        if not transcription:
            raise Exception("Speech to text conversion failed")
        
        # Step 3: Translation
        translation_result = translate_text(transcription['text'], target_language)
        if not translation_result:
            raise Exception("Translation failed")
        
        # Step 4: Text to Speech
        tts = TextToSpeech(elevenlabs_api_key)
        translated_speech = tts.text_to_speech(
            translation_result['translated_text'], 
            language=target_language
        )
        if not translated_speech:
            raise Exception("Text to speech conversion failed")
        
        # Step 5: Audio Insertion & Video Generation
        final_video = synchronize_and_generate_video(input_video_path, translated_speech)
        if not final_video:
            raise Exception("Final video generation failed")
        
        return final_video
    
    except Exception as e:
        print(f"Translation workflow error: {e}")
        return None

def main():
    # Load environment variables
    load_dotenv()
    
    # Get ElevenLabs API Key from environment
    elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    if not elevenlabs_api_key:
        print("Error: ELEVENLABS_API_KEY not found in .env file")
        sys.exit(1)
    
    # Prompt for input
    while True:
        input_source = input("Enter the path to the video file or YouTube URL: ").strip()
        
        # Check if it's a YouTube URL
        if is_youtube_url(input_source):
            print("YouTube URL detected. Attempting to download video...")
            input_video = download_youtube_video(input_source)
            
            if not input_video:
                print("Failed to download YouTube video. Please try another URL or a local file.")
                continue
            break
        
        # Check if it's a local file
        elif os.path.isfile(input_source):
            input_video = input_source
            break
        
        else:
            print("Invalid input. Please provide a valid local video file path or YouTube URL.")
    
    # Prompt for target language
    target_language = input("Enter the target language code (e.g., 'fr' for French): ")
    
    # Translate video
    final_video = translate_video(input_video, target_language, elevenlabs_api_key)
    
    if final_video:
        print(f"Translation complete. Final video: {final_video}")
        print("The final translated video has been saved and will be kept for future reference.")
    else:
        print("Video translation failed.")

if __name__ == "__main__":
    main()