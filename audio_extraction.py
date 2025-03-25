import subprocess
import speech_recognition as sr

def extract_audio(input_video, output_audio='input_audio.wav'):
    # Extract audio using ffmpeg
    subprocess.run(['ffmpeg', '-i', input_video, '-q:a', '0', '-map', 'a', output_audio], check=True)
    print("Audio extracted:", output_audio)

video_file = 'input_video.mp4'
extract_audio(video_file)