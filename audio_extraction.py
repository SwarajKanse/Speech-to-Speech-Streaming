import os
import ffmpeg

# Ensure the temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_audio(video_path):
    """
    Extract audio from a video file using FFmpeg.
    
    Args:
        video_path (str): Path to the input video file
    
    Returns:
        str: Path to the extracted audio file
    """
    try:
        # Generate output audio filename
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        audio_filename = f"{video_filename}_audio.wav"
        audio_path = os.path.join(TEMP_DIR, audio_filename)
        
        # Use FFmpeg to extract audio
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        print(f"Audio extracted successfully: {audio_path}")
        return audio_path
    
    except ffmpeg.Error as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def main():
    # Example usage
    video_path = input("Enter the path to the video file: ")
    extracted_audio = extract_audio(video_path)
    
    if extracted_audio:
        print(f"Audio extracted to: {extracted_audio}")
    else:
        print("Audio extraction failed.")

if __name__ == "__main__":
    main()