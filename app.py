import os
import uuid
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import main  # Import your main translation script
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
# More comprehensive CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type", 
            "Authorization", 
            "Access-Control-Allow-Credentials"
        ],
        "supports_credentials": True
    }
})

# Ensure temp directory exists
TEMP_DIR = 'temp'
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/api/translate-video', methods=['POST', 'OPTIONS'])
def translate_video():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response

    try:
        # Handle video input (file upload or YouTube URL)
        video_path = None
        
        # Check for file upload
        if 'video' in request.files:
            video_file = request.files['video']
            # Generate unique filename
            filename = f"{uuid.uuid4()}_{video_file.filename}"
            video_path = os.path.join(TEMP_DIR, filename)
            video_file.save(video_path)
        
        # Check for YouTube URL
        elif 'youtube_url' in request.form:
            youtube_url = request.form['youtube_url']
            video_path = main.download_youtube_video(youtube_url)
        
        # Check if video was successfully obtained
        if not video_path:
            return jsonify({"error": "No video provided"}), 400
        
        # Get target language
        target_language = request.form.get('target_language', 'en')
        
        # Get ElevenLabs API Key from environment
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            return jsonify({"error": "ElevenLabs API Key not found"}), 500
        
        # Translate video using your main script
        final_video = main.translate_video(
            video_path, 
            target_language, 
            elevenlabs_api_key
        )
        
        if not final_video:
            return jsonify({"error": "Video translation failed"}), 500
        
        # Explicitly use the correct path
        final_video_path = os.path.join(TEMP_DIR, 'final_translated_video.mp4')
        
        # Double-check file existence
        if not os.path.exists(final_video_path):
            print(f"ERROR: Final video path does not exist: {final_video_path}")
            print(f"Contents of temp directory: {os.listdir(TEMP_DIR)}")
            return jsonify({"error": "Translated video file not found"}), 500
        
        # Create response with CORS headers
        response = send_file(
            final_video_path, 
            mimetype='video/mp4', 
            as_attachment=True, 
            download_name='translated_video.mp4'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    except Exception as e:
        print(f"Translation error: {e}")
        error_response = jsonify({"error": str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)