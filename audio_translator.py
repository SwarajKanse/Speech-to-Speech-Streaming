import os
import tempfile
import logging
import time
import argparse
from pathlib import Path
import concurrent.futures
import gc
import threading
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
from dotenv import load_dotenv
import re
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from TTS.api import TTS
from google.generativeai import configure, GenerativeModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OptimizedAudioTranslator:
    def __init__(self, output_path=None, chunk_size=250, cleanup_temp=True, 
                 whisper_model_size="medium", use_gpu=True):
        """
        Initialize the OptimizedAudioTranslator with configuration options.
        
        Args:
            output_path (str): Directory to save output files. Default is "output_audio".
            chunk_size (int): Maximum character length for TTS chunks. Default is 250.
            cleanup_temp (bool): Whether to clean up temporary files. Default is True.
            whisper_model_size (str): Size of the Whisper model. Default is "medium".
            use_gpu (bool): Whether to use GPU for processing. Default is True.
        """
        # Load environment variables
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        # Set up output directory
        self.output_path = output_path or "output_audio"
        os.makedirs(self.output_path, exist_ok=True)
        
        # Configuration
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.compute_type = "float16" if self.use_gpu else "int8"
        self.chunk_size = chunk_size
        self.cleanup_temp = cleanup_temp
        self.temp_files = []
        self.whisper_model_size = whisper_model_size
        
        # Initialize models lazily (they'll be loaded when needed)
        self._whisper_model = None
        self._tts_model = None
        self._gemini_model = None
        
        logger.info(f"Initialized OptimizedAudioTranslator (device: {self.device}, "
                   f"whisper model: {whisper_model_size}, compute type: {self.compute_type})")
    
    @property
    def whisper_model(self):
        """Lazy loading of the Whisper model using faster-whisper."""
        if self._whisper_model is None:
            logger.info(f"Loading faster-whisper {self.whisper_model_size} model...")
            start_time = time.time()
            
            # Use GPU if available, with appropriate compute type
            self._whisper_model = WhisperModel(
                self.whisper_model_size, 
                device=self.device,
                compute_type=self.compute_type,
                download_root="./models"  # Cache models locally
            )
            
            logger.info(f"Whisper model loaded in {time.time() - start_time:.2f} seconds")
        return self._whisper_model
    
    @property
    def tts_model(self):
        """Lazy loading of the TTS model."""
        if self._tts_model is None:
            logger.info(f"Loading XTTS model on {self.device}...")
            start_time = time.time()
            self._tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info(f"XTTS model loaded in {time.time() - start_time:.2f} seconds")
        return self._tts_model
    
    @property
    def gemini_model(self):
        """Lazy loading of the Gemini model."""
        if self._gemini_model is None:
            logger.info("Configuring Gemini model...")
            configure(api_key=self.gemini_api_key)
            self._gemini_model = GenerativeModel("gemini-1.5-flash")
        return self._gemini_model
    
    def _clear_memory(self):
        """Clear CUDA memory cache."""
        if self.use_gpu:
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA memory cache")
    
    def convert_to_wav(self, input_path):
        """
        Convert audio to WAV format.
        
        Args:
            input_path (str): Path to the input audio file.
            
        Returns:
            str: Path to the converted WAV file.
        """
        try:
            logger.info(f"Converting {input_path} to WAV format")
            audio = AudioSegment.from_file(input_path)
            
            # Create a temporary WAV file
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            audio.export(wav_path, format="wav")
            self.temp_files.append(wav_path)
            return wav_path
        except Exception as e:
            raise ValueError(f"Failed to convert audio file: {e}")
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio for better transcription quality.
        
        Args:
            audio_path (str): Path to the WAV audio file.
            
        Returns:
            str: Path to the preprocessed audio file.
        """
        try:
            logger.info("Preprocessing audio for improved transcription...")
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize volume for better transcription
            audio = audio.normalize()
            
            # Remove silence to speed up processing
            # Keep 300ms of silence at the beginning and end
            audio_chunks = split_on_silence(
                audio, 
                min_silence_len=500,  # Minimum silence length (ms)
                silence_thresh=-40,   # Silence threshold (dB)
                keep_silence=300      # Keep 300ms of silence
            )
            
            # Combine chunks with a small amount of silence between
            processed_audio = AudioSegment.empty()
            for chunk in audio_chunks:
                processed_audio += chunk + AudioSegment.silent(duration=100)
            
            # Export to a temporary file
            fd, processed_path = tempfile.mkstemp(suffix="_processed.wav")
            os.close(fd)
            processed_audio.export(processed_path, format="wav")
            self.temp_files.append(processed_path)
            
            logger.info(f"Audio preprocessing complete. Duration reduced from "
                       f"{len(audio)/1000:.2f}s to {len(processed_audio)/1000:.2f}s")
            return processed_path
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}. Using original audio.")
            return audio_path
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using faster-whisper.
        
        Args:
            audio_path (str): Path to the WAV audio file.
            
        Returns:
            str: Transcribed text.
        """
        logger.info("Transcribing audio with faster-whisper...")
        start_time = time.time()
        
        # Preprocess the audio for better quality
        processed_audio = self.preprocess_audio(audio_path)
        
        # Transcribe with optimized settings
        segments, info = self.whisper_model.transcribe(
            processed_audio,
            beam_size=5,           # Higher beam size for better accuracy
            language=None,         # Auto-detect language
            vad_filter=True,       # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)  # Filter silence
        )
        
        # Collect all segments
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
        
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Clean up memory
        if self.use_gpu:
            self._clear_memory()
            
        return transcribed_text.strip()
    
    def translate_text(self, text, target_lang="hi", source_lang=None):
        """
        Translate text using a two-tier approach for optimal speed and accuracy.
        
        Args:
            text (str): Text to translate.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            source_lang (str): Source language code. Default is None (auto-detect).
            
        Returns:
            str: Translated text.
        """
        logger.info(f"Translating to '{target_lang}'...")
        start_time = time.time()
        
        # Skip translation if source and target are the same
        if source_lang and source_lang == target_lang:
            logger.info("Source and target languages are the same, skipping translation")
            return text
            
        # Split very long text for better translation quality
        if len(text) > 1500:
            return self._translate_long_text(text, target_lang, source_lang)
        
        try:
            # First try Google Translate (faster)
            translator = GoogleTranslator(source=source_lang or "auto", target=target_lang)
            translated = translator.translate(text)
            
            # Verify translation quality (if much shorter, might be poor quality)
            if len(translated) < len(text) * 0.5 and len(text) > 100:
                logger.warning("Translation appears too short, might be low quality")
                raise ValueError("Potentially low-quality translation, trying Gemini instead")
                
            translation_time = time.time() - start_time
            logger.info(f"Google Translate completed in {translation_time:.2f} seconds")
            return translated.strip()
            
        except Exception as e:
            logger.warning(f"Google Translate failed or produced low-quality result: {e}")
            logger.info("Trying Gemini for better accuracy...")
            
            try:
                gemini_start = time.time()
                
                # Create a more detailed prompt for Gemini
                prompt = f"""Please translate the following text to {target_lang}. 
                Maintain the original style, tone, and format:

                {text}
                
                Translation:"""
                
                response = self.gemini_model.generate_content(prompt)
                translated = response.text.strip()
                
                gemini_time = time.time() - gemini_start
                logger.info(f"Gemini translation completed in {gemini_time:.2f} seconds")
                return translated
                
            except Exception as e2:
                logger.error(f"Gemini translation also failed: {e2}")
                # Fall back to original Google Translate result if we have it
                if 'translated' in locals() and translated:
                    return translated.strip()
                return text  # Worst case, return original
    
    def _translate_long_text(self, text, target_lang, source_lang=None):
        """
        Translate long text by splitting it into paragraphs.
        
        Args:
            text (str): Long text to translate.
            target_lang (str): Target language code.
            source_lang (str): Source language code. Default is None (auto-detect).
            
        Returns:
            str: Combined translated text.
        """
        logger.info("Splitting long text for optimal translation")
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split extremely long paragraphs
        split_paragraphs = []
        for para in paragraphs:
            if len(para) > 1000:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sentence in sentences:
                    if len(current) + len(sentence) < 1000:
                        current += sentence + " "
                    else:
                        if current:
                            split_paragraphs.append(current.strip())
                        current = sentence + " "
                if current:
                    split_paragraphs.append(current.strip())
            else:
                split_paragraphs.append(para)
        
        # Translate each part
        translated_parts = []
        for i, part in enumerate(split_paragraphs):
            logger.info(f"Translating part {i+1}/{len(split_paragraphs)}")
            translated = self.translate_text(part, target_lang, source_lang)
            translated_parts.append(translated)
        
        # Join with double newlines to preserve paragraph structure
        return "\n\n".join(translated_parts)
    
    def smart_split_text(self, text):
        """
        Split text into chunks without cutting sentences.
        
        Args:
            text (str): Text to split.
            
        Returns:
            list: List of text chunks.
        """
        # This regex covers punctuation for multiple languages
        sentence_endings = r'[.!?।؛؟\n]'
        
        # Split while keeping the punctuation
        parts = re.split(f'({sentence_endings})', text)
        
        # Combine each sentence with its punctuation
        sentences = []
        for i in range(0, len(parts)-1, 2):
            if i+1 < len(parts):
                sentence = parts[i].strip() + parts[i+1].strip()
                if sentence:
                    sentences.append(sentence)
        
        # Handle trailing text without punctuation
        if len(parts) % 2 != 0 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding the sentence exceeds the chunk_size, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    # If the sentence itself is longer than chunk_size, hard-split it
                    while len(sentence) > self.chunk_size:
                        chunks.append(sentence[:self.chunk_size])
                        sentence = sentence[self.chunk_size:]
                    current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Text split into {len(chunks)} chunk(s)")
        return chunks
    
    def compute_speaker_embedding(self, input_audio):
        """
        Compute speaker embedding once to reuse across TTS chunks.
        
        Args:
            input_audio (str): Path to the reference voice audio file.
            
        Returns:
            Any: Speaker embedding object that can be reused.
        """
        logger.info("Computing speaker embedding...")
        start_time = time.time()
        
        # Ensure TTS model is loaded
        tts_model = self.tts_model
        
        # Compute the embedding
        speaker_embedding = tts_model.synthesizer.tts_model.speaker_manager.compute_embedding(input_audio)
        
        logger.info(f"Speaker embedding computed in {time.time() - start_time:.2f} seconds")
        return speaker_embedding
    
    def clone_voice_xtts_optimized(self, input_audio, translated_text, target_lang="hi"):
        """
        Optimized voice cloning using pre-computed speaker embedding.
        
        Args:
            input_audio (str): Path to the reference voice audio file.
            translated_text (str): Translated text to synthesize.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            tuple: Tuple containing (list of audio part paths, combined audio path).
        """
        logger.info("Cloning voice with optimized XTTS...")
        text_chunks = self.smart_split_text(translated_text)
        
        # Pre-compute speaker embedding once
        speaker_embedding = self.compute_speaker_embedding(input_audio)
        
        # Process chunks sequentially with the same model instance
        cloned_audio_paths = []
        
        for i, chunk in enumerate(text_chunks):
            try:
                logger.info(f"Synthesizing part {i+1}/{len(text_chunks)}...")
                start_time = time.time()
                
                part_path = os.path.join(self.output_path, f"cloned_audio_part_{i + 1}.wav")
                
                # Use pre-computed speaker embedding
                self.tts_model.tts_to_file(
                    text=chunk,
                    file_path=part_path,
                    speaker_wav=None,  # Don't recompute embedding
                    speaker_embedding=speaker_embedding,
                    language=target_lang,
                )
                
                synthesis_time = time.time() - start_time
                logger.info(f"Part {i+1} synthesized in {synthesis_time:.2f} seconds")
                cloned_audio_paths.append(part_path)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
        
        # Combine all audio parts
        combined_path = os.path.join(self.output_path, "translated_audio_combined.wav")
        if len(cloned_audio_paths) > 1:
            self._combine_audio_files(cloned_audio_paths, combined_path)
        elif cloned_audio_paths:
            combined_path = cloned_audio_paths[0]
        else:
            logger.error("No audio parts were successfully generated")
            combined_path = None
        
        # Clear memory
        self._clear_memory()
            
        return cloned_audio_paths, combined_path
    
    def _combine_audio_files(self, audio_paths, output_path):
        """
        Combine multiple audio files into one.
        
        Args:
            audio_paths (list): List of paths to audio files.
            output_path (str): Path to save the combined audio.
        """
        logger.info(f"Combining {len(audio_paths)} audio parts...")
        combined = AudioSegment.empty()
        
        for path in audio_paths:
            segment = AudioSegment.from_file(path)
            combined += segment
            
        # Normalize volume of final audio
        combined = combined.normalize()
        
        combined.export(output_path, format="wav")
        logger.info(f"Combined audio saved to: {output_path}")
    
    def translate_and_clone(self, input_audio, target_lang="hi"):
        """
        Full pipeline: Transcribe, Translate, and Clone Voice with optimizations.
        
        Args:
            input_audio (str): Path to the input audio file.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            dict: Results including original text, translated text, and audio paths.
        """
        logger.info(f"🚀 Starting optimized translation and cloning to {target_lang}...")
        total_start_time = time.time()
        
        # Step 1: Convert audio to WAV
        input_wav = self.convert_to_wav(input_audio)
        logger.info(f"🎵 Converted audio to WAV: {input_wav}")
        
        # Step 2: Transcribe with whisper
        original_text = self.transcribe_audio(input_wav)
        logger.info(f"📄 Original Text: {original_text[:100]}...")
        
        # Get the detected language from the transcription (for possible skipping translation)
        detected_lang = self._whisper_model.detect_language(input_wav)[0]
        logger.info(f"Detected language: {detected_lang}")
        
        # Unload Whisper model to free memory
        self._whisper_model = None
        self._clear_memory()
        
        # Step 3: Translate
        translated_text = self.translate_text(original_text, target_lang, source_lang=detected_lang)
        logger.info(f"🌐 Translated Text: {translated_text[:100]}...")
        
        # Step 4: Clone voice with optimized approach
        audio_parts, combined_audio = self.clone_voice_xtts_optimized(input_wav, translated_text, target_lang)
        
        # Unload TTS model to free memory
        self._tts_model = None
        self._clear_memory()
        
        total_time = time.time() - total_start_time
        logger.info(f"✅ Done! Processing completed in {total_time:.2f} seconds")
        
        # Clean up temporary files if requested
        if self.cleanup_temp:
            self._cleanup()
        
        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "audio_parts": audio_parts,
            "combined_audio": combined_audio,
            "processing_time_seconds": total_time
        }
    
    def _cleanup(self):
        """Clean up temporary files."""
        logger.info(f"Cleaning up {len(self.temp_files)} temporary files")
        for file_path in self.temp_files:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        self.temp_files = []
    
    def unload_models(self):
        """Manually unload all models to free memory."""
        logger.info("Unloading all models to free memory")
        self._whisper_model = None
        self._tts_model = None
        self._gemini_model = None
        self._clear_memory()


class StreamingAudioTranslator(OptimizedAudioTranslator):
    """
    Enhanced version that implements streaming pipeline processing.
    Each step starts as soon as data is available from the previous step.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_queue = {}
        self._lock = threading.Lock()
    
    def _transcription_worker(self, job_id, input_audio, callback=None):
        """Worker thread for transcription."""
        try:
            # Convert and transcribe
            input_wav = self.convert_to_wav(input_audio)
            transcribed_text = self.transcribe_audio(input_wav)
            
            # Store results
            with self._lock:
                self.results_queue[job_id] = {
                    "input_wav": input_wav,
                    "transcribed_text": transcribed_text,
                    "status": "transcribed"
                }
            
            # Callback if provided
            if callback:
                callback(job_id, "transcription", transcribed_text)
            
            # Unload whisper model
            self._whisper_model = None
            self._clear_memory()
            
            # Start translation
            threading.Thread(
                target=self._translation_worker, 
                args=(job_id, transcribed_text, callback)
            ).start()
            
        except Exception as e:
            logger.error(f"Transcription error for job {job_id}: {e}")
            with self._lock:
                self.results_queue[job_id]["status"] = "error"
                self.results_queue[job_id]["error"] = str(e)
    
    def _translation_worker(self, job_id, text, callback=None, target_lang="hi"):
        """Worker thread for translation."""
        try:
            # Translate
            translated_text = self.translate_text(text, target_lang)
            
            # Store results
            with self._lock:
                self.results_queue[job_id]["translated_text"] = translated_text
                self.results_queue[job_id]["status"] = "translated"
            
            # Callback if provided
            if callback:
                callback(job_id, "translation", translated_text)
            
            # Start voice cloning
            input_wav = self.results_queue[job_id]["input_wav"]
            
            threading.Thread(
                target=self._tts_worker, 
                args=(job_id, input_wav, translated_text, target_lang, callback)
            ).start()
            
        except Exception as e:
            logger.error(f"Translation error for job {job_id}: {e}")
            with self._lock:
                self.results_queue[job_id]["status"] = "error"
                self.results_queue[job_id]["error"] = str(e)
    
    def _tts_worker(self, job_id, input_wav, translated_text, target_lang, callback=None):
        """Worker thread for TTS."""
        try:
            # Clone voice
            audio_parts, combined_audio = self.clone_voice_xtts_optimized(
                input_wav, translated_text, target_lang
            )
            
            # Store results
            with self._lock:
                self.results_queue[job_id]["audio_parts"] = audio_parts
                self.results_queue[job_id]["combined_audio"] = combined_audio
                self.results_queue[job_id]["status"] = "completed"
            
            # Callback if provided
            if callback:
                callback(job_id, "tts", combined_audio)
            
            # Unload TTS model
            self._tts_model = None
            self._clear_memory()
            
        except Exception as e:
            logger.error(f"TTS error for job {job_id}: {e}")
            with self._lock:
                self.results_queue[job_id]["status"] = "error"
                self.results_queue[job_id]["error"] = str(e)
    
    def process_streaming(self, input_audio, target_lang="hi", callback=None):
        """
        Process audio with a streaming pipeline architecture.
        
        Args:
            input_audio (str): Path to the input audio file.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            callback (callable): Function to call with updates. Takes (job_id, stage, data).
            
        Returns:
            str: Job ID for tracking the process.
        """
        # Generate job ID
        job_id = f"job_{int(time.time())}"
        
        # Initialize job
        with self._lock:
            self.results_queue[job_id] = {
                "status": "started",
                "start_time": time.time()
            }
        
        # Start the pipeline
        threading.Thread(
            target=self._transcription_worker, 
            args=(job_id, input_audio, callback)
        ).start()
        
        return job_id
    
    def get_job_status(self, job_id):
        """
        Get the status of a job.
        
        Args:
            job_id (str): Job ID.
            
        Returns:
            dict: Job status and results.
        """
        with self._lock:
            if job_id in self.results_queue:
                return self.results_queue[job_id]
            else:
                return {"status": "not_found"}


def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description="Optimized Audio Translation Pipeline")
    parser.add_argument("input_audio", help="Path to the input audio file")
    parser.add_argument("--target-lang", "-t", default="hi", help="Target language code (default: hi)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: output_audio)")
    parser.add_argument("--chunk-size", "-c", type=int, default=250, 
                        help="Maximum character length for TTS chunks (default: 250)")
    parser.add_argument("--whisper-model", "-w", default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU processing")
    parser.add_argument("--streaming", action="store_true", help="Use streaming pipeline architecture")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.streaming:
        # Use streaming architecture
        translator = StreamingAudioTranslator(
            output_path=args.output_dir,
            chunk_size=args.chunk_size,
            cleanup_temp=not args.keep_temp,
            whisper_model_size=args.whisper_model,
            use_gpu=not args.no_gpu
        )
        
        # Define callback function to display progress
        def update_callback(job_id, stage, data):
            if stage == "transcription":
                print(f"\n--- Transcription Completed ---")
                print(f"Text: {data[:100]}...")
            elif stage == "translation":
                print(f"\n--- Translation Completed ---")
                print(f"Text: {data[:100]}...")
            elif stage == "tts":
                print(f"\n--- Voice Cloning Completed ---")
                print(f"Audio saved to: {data}")
        
        # Start streaming process
        job_id = translator.process_streaming(args.input_audio, args.target_lang, update_callback)
        print(f"Started streaming job with ID: {job_id}")
        
        # Wait for job to complete
        while True:
            status = translator.get_job_status(job_id)
            if status["status"] in ["completed", "error"]:
                break
            time.sleep(1)
            
        # Display final results
        if status["status"] == "completed":
            print("\n--- Final Results ---")
            print(f"Original text: {status['transcribed_text'][:100]}...")
            print(f"Translated text: {status['translated_text'][:100]}...")
            print(f"Combined audio: {status['combined_audio']}")
            print(f"Processing time: {time.time() - status['start_time']:.2f} seconds")
        else:
            print(f"\nError: {status.get('error', 'Unknown error')}")
            
    else:
        # Use standard architecture
        translator = OptimizedAudioTranslator(
            output_path=args.output_dir,
            chunk_size=args.chunk_size,
            cleanup_temp=not args.keep_temp,
            whisper_model_size=args.whisper_model,
            use_gpu=not args.no_gpu
        )
        
        try:
            start_time = time.time()
            results = translator.translate_and_clone(args.input_audio, args.target_lang)
            
            print("\n--- Results ---")
            print(f"Original text: {results['original_text'][:100]}...")
            print(f"Translated text: {results['translated_text'][:100]}...")
            print(f"Combined audio: {results['combined_audio']}")
            print(f"Processing time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during processing: {e}")