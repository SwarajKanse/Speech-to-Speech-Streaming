import os
import sys
import subprocess

def extract_audio(input_video=None, output_audio='input_audio.wav'):
    """
    Extract audio from a video file using ffmpeg.
    
    Args:
    input_video (str, optional): Path to the input video file.
    output_audio (str, optional): Name of the output audio file.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Full path for output audio in the same directory as the script
    output_path = os.path.join(script_dir, output_audio)
    
    # If no input video is provided, prompt user
    while not input_video or not os.path.exists(input_video):
        if input_video:
            print(f"Error: File '{input_video}' not found.")
        
        input_video = input("Please enter the full path to your video file: ").strip()
        # Remove quotes if user accidentally included them
        input_video = input_video.strip("'\"")
    
    try:
        # Extract audio using ffmpeg
        subprocess.run(['ffmpeg', '-i', input_video, '-q:a', '0', '-map', 'a', output_path], 
                       check=True, 
                       stderr=subprocess.PIPE)
        
        print(f"Audio successfully extracted to: {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        print("Error extracting audio:")
        print(e.stderr.decode() if e.stderr else "Unknown error occurred")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main():
    # Allow passing video file as command-line argument
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = None
    
    # Extract audio
    extracted_audio = extract_audio(video_file)
    
    if extracted_audio:
        print(f"Audio extraction complete. File saved as: {extracted_audio}")

if __name__ == "__main__":
    main()