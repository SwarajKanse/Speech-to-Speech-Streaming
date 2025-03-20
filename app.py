# app.py - Main Flask application
from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import torch
from werkzeug.utils import secure_filename
import whisper
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import cv2
import numpy as np
from deepface import DeepFace
from resemblyzer import VoiceEncoder, preprocess_wav
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import moviepy.editor as mp
from pydub import AudioSegment
import tempfile
import librosa
from TTS.api import TTS
import ffmpeg
import yt_dlp
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
print("Loading models...")
# Speech recognition model
whisper_model = whisper.load_model("medium")
# Translation model
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# Face detection model
face_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
face_detection_model.eval()
# Voice embedding model
voice_encoder = VoiceEncoder()
# Text-to-speech model
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Language code mapping
LANGUAGE_CODES = {
    "English": "en_XX",
    "Spanish": "es_XX",
    "French": "fr_XX",
    "German": "de_DE",
    "Chinese": "zh_CN",
    "Japanese": "ja_XX",
    "Korean": "ko_KR",
    "Russian": "ru_RU",
    "Hindi": "hi_IN",
    "Arabic": "ar_AR",
    "Italian": "it_IT",
    "Portuguese": "pt_XX",
}

# Reverse mapping for TTS
TTS_LANG_CODES = {
    "en_XX": "en",
    "es_XX": "es",
    "fr_XX": "fr",
    "de_DE": "de",
    "zh_CN": "zh-cn",
    "ja_XX": "ja",
    "ko_KR": "ko",
    "ru_RU": "ru",
    "hi_IN": "hi",
    "ar_AR": "ar",
    "it_IT": "it", 
    "pt_XX": "pt"
}

def download_youtube_video(youtube_url):
    """Download a YouTube video and return the path to the saved file."""
    video_id = str(uuid.uuid4())
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return output_path

def extract_audio(video_path):
    """Extract audio from video file."""
    audio_path = video_path.replace('.mp4', '.wav')
    video_clip = mp.VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path

def transcribe_audio(audio_path, source_lang="en"):
    """Transcribe audio file using Whisper."""
    # Transcribe with timestamps
    result = whisper_model.transcribe(
        audio_path, 
        verbose=False,
        task="transcribe",
        language=source_lang[:2]  # Whisper uses 2-letter codes
    )
    
    # Extract segments with timestamps
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    return segments

def translate_text(text, target_lang):
    """Translate text using MBart."""
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    src_lang = "en_XX"  # Default source language
    
    # Tokenize and translate
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    
    # Decode the translated text
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def detect_faces_in_video(video_path):
    """
    Detect and track faces throughout the video.
    Returns a dictionary mapping face IDs to their appearance timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    face_tracks = {}
    current_faces = {}
    next_face_id = 0
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 5th frame for efficiency
        if frame_count % 5 == 0:
            # Convert frame to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            try:
                faces = DeepFace.extract_faces(rgb_frame, detector_backend='opencv')
                detected_faces = []
                
                for face_data in faces:
                    if face_data['confidence'] > 0.9:  # Filter by confidence
                        face_img = face_data['face']
                        face_embedding = DeepFace.represent(face_img, model_name="Facenet")[0]["embedding"]
                        detected_faces.append(face_embedding)
                
                # Update face tracks
                new_current_faces = {}
                
                # Try to match with existing faces
                for face_embedding in detected_faces:
                    matched = False
                    
                    for face_id, existing_embedding in current_faces.items():
                        # Calculate similarity (Euclidean distance)
                        similarity = np.linalg.norm(np.array(face_embedding) - np.array(existing_embedding))
                        
                        if similarity < 0.6:  # Threshold for considering same face
                            matched = True
                            new_current_faces[face_id] = face_embedding
                            
                            # Update timestamps for this face
                            if face_id not in face_tracks:
                                face_tracks[face_id] = []
                            
                            timestamp = frame_count / fps
                            face_tracks[face_id].append(timestamp)
                            break
                    
                    if not matched:
                        # New face detected
                        face_id = next_face_id
                        next_face_id += 1
                        new_current_faces[face_id] = face_embedding
                        
                        # Create new entry in face_tracks
                        face_tracks[face_id] = [frame_count / fps]
                
                current_faces = new_current_faces
            except:
                # No faces detected in this frame
                pass
                
        frame_count += 1
    
    cap.release()
    
    # Consolidate face appearances into time ranges
    face_segments = {}
    for face_id, timestamps in face_tracks.items():
        timestamps.sort()
        segments = []
        current_segment = [timestamps[0], timestamps[0]]
        
        for ts in timestamps[1:]:
            # If this timestamp is close to the end of current segment, extend the segment
            if ts - current_segment[1] < 0.5:  # 0.5 second threshold
                current_segment[1] = ts
            else:
                # Otherwise, close current segment and start a new one
                segments.append(current_segment)
                current_segment = [ts, ts]
        
        segments.append(current_segment)
        face_segments[face_id] = segments
    
    return face_segments

def analyze_speaker_audio(audio_path, face_segments, transcribed_segments):
    """
    Analyze audio to detect different speakers and match them with face segments.
    Returns the transcribed segments with speaker IDs.
    """
    # Load audio file
    wav, sr = librosa.load(audio_path, sr=16000)
    
    # Speaker mapping - match face segments with audio segments
    speaker_map = {}
    
    # Go through each transcribed segment
    for segment in transcribed_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        mid_time = (start_time + end_time) / 2
        
        # Find faces visible during this segment
        visible_faces = []
        for face_id, face_time_ranges in face_segments.items():
            for time_range in face_time_ranges:
                if start_time <= time_range[1] and end_time >= time_range[0]:
                    visible_faces.append(face_id)
                    break
        
        # If only one face is visible during speech, assign that face as speaker
        if len(visible_faces) == 1:
            segment["speaker_id"] = visible_faces[0]
        else:
            # Otherwise, try to determine speaker from audio characteristics
            # Extract audio for this segment
            segment_samples = wav[int(start_time * sr):int(end_time * sr)]
            
            if len(segment_samples) > 0:
                # Only process if we have enough audio
                if len(segment_samples) > sr * 0.5:  # At least 0.5 seconds
                    # Preprocess for speaker embedding
                    segment_preprocessed = preprocess_wav(segment_samples, sr)
                    # Get speaker embedding
                    if len(segment_preprocessed) > 0:
                        speaker_embedding = voice_encoder.embed_utterance(segment_preprocessed)
                        
                        # If we've seen this speaker before, find the closest match
                        if len(speaker_map) > 0:
                            best_match = None
                            best_similarity = -1
                            
                            for speaker_id, embedding in speaker_map.items():
                                similarity = np.dot(speaker_embedding, embedding)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = speaker_id
                            
                            if best_similarity > 0.75:  # Threshold for considering same speaker
                                segment["speaker_id"] = best_match
                            else:
                                # New speaker
                                if len(visible_faces) > 0:
                                    # Assign to one of the visible faces
                                    new_speaker_id = visible_faces[0]
                                    speaker_map[new_speaker_id] = speaker_embedding
                                    segment["speaker_id"] = new_speaker_id
                                else:
                                    # No faces visible, create new speaker ID
                                    new_speaker_id = f"speaker_{len(speaker_map)}"
                                    speaker_map[new_speaker_id] = speaker_embedding
                                    segment["speaker_id"] = new_speaker_id
                        else:
                            # First speaker
                            if len(visible_faces) > 0:
                                # Assign to one of the visible faces
                                new_speaker_id = visible_faces[0]
                                speaker_map[new_speaker_id] = speaker_embedding
                                segment["speaker_id"] = new_speaker_id
                            else:
                                # No faces visible
                                new_speaker_id = "speaker_0"
                                speaker_map[new_speaker_id] = speaker_embedding
                                segment["speaker_id"] = new_speaker_id
                    else:
                        # Not enough data for speaker embedding
                        segment["speaker_id"] = "unknown"
                else:
                    # Segment too short
                    segment["speaker_id"] = "unknown"
            else:
                # No audio for this segment
                segment["speaker_id"] = "unknown"
    
    return transcribed_segments

def generate_translated_audio(segments, target_lang, speaker_count=2):
    """
    Generate translated audio for each segment with appropriate voice for each speaker.
    Returns a list of audio file paths with timing information.
    """
    audio_segments = []
    
    # Get unique speaker IDs
    speakers = set()
    for segment in segments:
        if "speaker_id" in segment:
            speakers.add(segment["speaker_id"])
    
    # Map speakers to voice styles
    voice_styles = ["p326", "p330", "p339", "p362", "p361", "p225"]
    speaker_voices = {}
    
    for i, speaker in enumerate(speakers):
        voice_idx = i % len(voice_styles)
        speaker_voices[speaker] = voice_styles[voice_idx]
    
    # Generate audio for each segment
    for i, segment in enumerate(segments):
        # Select voice based on speaker
        if "speaker_id" in segment and segment["speaker_id"] in speaker_voices:
            voice = speaker_voices[segment["speaker_id"]]
        else:
            # Default voice if speaker is unknown
            voice = voice_styles[0]
        
        # Translate text
        translated_text = translate_text(segment["text"], target_lang)
        
        # Clean text for TTS
        cleaned_text = re.sub(r'[^\w\s.,!?\'"-]', '', translated_text)
        
        # Generate audio
        tts_lang_code = TTS_LANG_CODES.get(target_lang, "en")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"segment_{i}.wav")
        
        try:
            tts_model.tts_to_file(
                text=cleaned_text,
                file_path=output_path,
                speaker=voice,
                language=tts_lang_code
            )
            
            # Adjust duration to match original segment if needed
            original_duration = segment["end"] - segment["start"]
            
            # Load generated audio
            audio = AudioSegment.from_file(output_path)
            generated_duration = len(audio) / 1000.0  # Convert ms to seconds
            
            # Add to list with timing info
            audio_segments.append({
                "path": output_path,
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text,
                "speaker_id": segment.get("speaker_id", "unknown"),
                "original_duration": original_duration,
                "generated_duration": generated_duration
            })
            
        except Exception as e:
            print(f"Error generating audio for segment {i}: {e}")
            # Use empty audio as fallback
            audio_segments.append({
                "path": None,
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text,
                "speaker_id": segment.get("speaker_id", "unknown"),
                "original_duration": segment["end"] - segment["start"],
                "generated_duration": 0
            })
    
    return audio_segments

def create_translated_video(video_path, audio_segments, output_path, target_lang):
    """
    Create a new video with translated audio and subtitles.
    """
    # Extract video information
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    
    # Create temporary subtitle file
    subtitle_path = os.path.join(app.config['UPLOAD_FOLDER'], "subtitles.srt")
    with open(subtitle_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(audio_segments):
            start_time_str = format_time_srt(segment["start"])
            end_time_str = format_time_srt(segment["end"])
            
            f.write(f"{i+1}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{segment['text']}\n\n")
    
    # Create concatenated audio file
    concat_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "concat_audio.wav")
    create_concat_audio(audio_segments, concat_audio_path, video_path)
    
    # Check if audio was successfully created
    if not os.path.exists(concat_audio_path) or os.path.getsize(concat_audio_path) == 0:
        # Fallback: use original audio
        concat_audio_path = extract_audio(video_path)
    
    # Combine video with new audio and subtitles
    try:
        # Input video
        video = ffmpeg.input(video_path)
        
        # Input audio
        audio = ffmpeg.input(concat_audio_path)
        
        # Combine video and audio
        output = ffmpeg.output(
            video.video,
            audio.audio,
            output_path,
            vcodec='copy',
            acodec='aac',
            vf=f"subtitles={subtitle_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=3'"
        )
        
        # Run ffmpeg
        output.run(overwrite_output=True, quiet=True)
        
        return output_path
    except Exception as e:
        print(f"Error creating translated video: {e}")
        return None

def format_time_srt(seconds):
    """Format time for SRT subtitles."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def create_concat_audio(audio_segments, output_path, video_path):
    """Create a concatenated audio file from segments with correct timing."""
    # Get video duration
    video = mp.VideoFileClip(video_path)
    video_duration = video.duration
    video.close()
    
    # Create a silent audio track matching video duration
    silent_audio = AudioSegment.silent(duration=int(video_duration * 1000))
    
    # Overlay each segment at the correct position
    for segment in audio_segments:
        if segment["path"] and os.path.exists(segment["path"]):
            segment_audio = AudioSegment.from_file(segment["path"])
            
            # Position in milliseconds
            start_pos = int(segment["start"] * 1000)
            
            # Overlay at the correct position
            silent_audio = silent_audio.overlay(segment_audio, position=start_pos)
    
    # Export to file
    silent_audio.export(output_path, format="wav")
    return output_path

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', languages=LANGUAGE_CODES)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload or YouTube URL."""
    try:
        target_lang = request.form.get('target_language')
        if not target_lang or target_lang not in LANGUAGE_CODES.values():
            return jsonify({'error': 'Invalid target language'}), 400
        
        source_lang = request.form.get('source_language', 'en')
        
        # Check if YouTube URL is provided
        youtube_url = request.form.get('youtube_url')
        if youtube_url:
            # Download YouTube video
            video_path = download_youtube_video(youtube_url)
        else:
            # Check if file is provided
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            # Save uploaded file
            filename = secure_filename(file.filename)
            video_id = str(uuid.uuid4())
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
            file.save(video_path)
        
        # Process the video (in a background task in production)
        # 1. Extract audio
        audio_path = extract_audio(video_path)
        
        # 2. Transcribe audio
        transcribed_segments = transcribe_audio(audio_path, source_lang)
        
        # 3. Detect faces in video
        face_segments = detect_faces_in_video(video_path)
        
        # 4. Analyze speakers in audio and match with faces
        speaker_segments = analyze_speaker_audio(audio_path, face_segments, transcribed_segments)
        
        # 5. Generate translated audio for each segment
        audio_segments = generate_translated_audio(speaker_segments, target_lang)
        
        # 6. Create translated video
        output_video_id = str(uuid.uuid4())
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{output_video_id}_translated.mp4")
        result_path = create_translated_video(video_path, audio_segments, output_path, target_lang)
        
        if result_path and os.path.exists(result_path):
            return jsonify({
                'success': True,
                'video_id': output_video_id,
                'message': 'Video processed successfully',
                'download_url': f'/download/{output_video_id}'
            })
        else:
            return jsonify({'error': 'Error processing video'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<video_id>')
def download(video_id):
    """Download processed video."""
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(f"{video_id}_")]
    if not files:
        return "File not found", 404
        
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], files[0]),
        as_attachment=True,
        attachment_filename="translated_video.mp4"
    )

if __name__ == '__main__':
    app.run(debug=True)