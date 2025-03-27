import os
import subprocess
import ffmpeg

# Ensure the temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def synchronize_and_generate_video(original_video_path, translated_audio_path):
    """
    Synchronize translated audio with original video length by adjusting audio speed.
    
    Args:
        original_video_path (str): Path to the original video file
        translated_audio_path (str): Path to the translated audio file
    
    Returns:
        str: Path to the generated video with synchronized audio
    """
    try:
        # Get original video duration
        probe = ffmpeg.probe(original_video_path)
        video_duration = float(probe['streams'][0]['duration'])
        
        # Get translated audio duration
        audio_probe = ffmpeg.probe(translated_audio_path)
        audio_duration = float(audio_probe['streams'][0]['duration'])
        
        # Prepare output video path
        output_video_path = os.path.join(TEMP_DIR, 'final_translated_video.mp4')
        
        # Prepare output synchronized audio path
        synchronized_audio_path = os.path.join(TEMP_DIR, 'speed_adjusted_audio.mp3')
        
        # Calculate speed modification factor
        # This will make the audio faster to fit the video duration
        speed_factor = audio_duration / video_duration
        
        # Use FFmpeg to adjust audio speed
        subprocess.run([
            'ffmpeg', 
            '-i', translated_audio_path, 
            '-filter:a', f'asetrate=44100*{speed_factor},aformat=sample_rates=44100', 
            '-c:a', 'libmp3lame', 
            synchronized_audio_path
        ], check=True)
        
        # Generate final video with speed-adjusted audio
        subprocess.run([
            'ffmpeg', 
            '-i', original_video_path, 
            '-i', synchronized_audio_path, 
            '-map', '0:v', 
            '-map', '1:a', 
            '-c:v', 'copy', 
            '-c:a', 'aac', 
            output_video_path
        ], check=True)
        
        print(f"Original Audio Duration: {audio_duration:.2f} seconds")
        print(f"Video Duration: {video_duration:.2f} seconds")
        print(f"Audio Speed Adjustment Factor: {speed_factor:.2f}")
        print(f"Video generated successfully: {output_video_path}")
        
        return output_video_path
    
    except Exception as e:
        print(f"Error in video generation: {e}")
        return None

def main():
    # Find original video in temp directory
    try:
        original_video = os.path.join(TEMP_DIR, [f for f in os.listdir(TEMP_DIR) if f.endswith('_video.mp4')][0])
    except IndexError:
        print("No original video found in the temp directory.")
        return

    # Find translated audio - now more flexible to match different naming patterns
    translated_audio_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('_speech.mp3') or f.startswith('translated_speech_')]
    
    if not translated_audio_files:
        print("No translated audio found in the temp directory.")
        return
    
    # Take the first matching audio file
    translated_audio = os.path.join(TEMP_DIR, translated_audio_files[0])
    
    # Generate synchronized video
    final_video = synchronize_and_generate_video(original_video, translated_audio)
    
    if final_video:
        print(f"Final translated video created: {final_video}")
    else:
        print("Failed to generate final video.")

if __name__ == "__main__":
    main()