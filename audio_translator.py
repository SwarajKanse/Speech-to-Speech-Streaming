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


class EnhancedAudioTranslator:
    def __init__(self, output_path=None, chunk_size=250, cleanup_temp=True, 
                 whisper_model_size="medium", use_gpu=True, batch_size=4):
        """
        Initialize the EnhancedAudioTranslator with configuration options.
        
        Args:
            output_path (str): Directory to save output files. Default is "output_audio".
            chunk_size (int): Maximum character length for TTS chunks. Default is 250.
            cleanup_temp (bool): Whether to clean up temporary files. Default is True.
            whisper_model_size (str): Size of the Whisper model. Default is "medium".
            use_gpu (bool): Whether to use GPU for processing. Default is True.
            batch_size (int): Batch size for parallel processing. Default is 4.
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
        self.batch_size = batch_size
        
        # Initialize models lazily (they'll be loaded when needed)
        self._whisper_model = None
        self._tts_model = None
        self._gemini_model = None
        
        # Cache for speaker embeddings
        self._speaker_embeddings = {}
        
        logger.info(f"Initialized EnhancedAudioTranslator (device: {self.device}, "
                   f"whisper model: {whisper_model_size}, compute type: {self.compute_type}, "
                   f"batch size: {batch_size})")
    
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
        Enhanced audio preprocessing for better transcription quality.
        
        Args:
            audio_path (str): Path to the WAV audio file.
            
        Returns:
            str: Path to the preprocessed audio file.
        """
        try:
            logger.info("Advanced preprocessing audio for improved transcription...")
            audio = AudioSegment.from_file(audio_path)
            
            # Step 1: Normalize volume for better transcription
            audio = audio.normalize()
            
            # Step 2: Apply a slight high-pass filter to reduce background noise
            # This improves voice clarity by reducing low-frequency noise
            audio = audio.high_pass_filter(80)
            
            # Step 3: Remove silence to speed up processing
            # Keep 300ms of silence at the beginning and end
            audio_chunks = split_on_silence(
                audio, 
                min_silence_len=500,  # Minimum silence length (ms)
                silence_thresh=-40,   # Silence threshold (dB)
                keep_silence=300      # Keep 300ms of silence
            )
            
            # Step 4: Combine chunks with a small amount of silence between for natural pauses
            processed_audio = AudioSegment.empty()
            for chunk in audio_chunks:
                processed_audio += chunk + AudioSegment.silent(duration=100)
            
            # Step 5: Boost speech frequencies slightly for better recognition
            # Human speech is typically in the 250-3000 Hz range
            processed_audio = processed_audio.low_pass_filter(3000)
            
            # Export to a temporary file
            fd, processed_path = tempfile.mkstemp(suffix="_processed.wav")
            os.close(fd)
            processed_audio.export(processed_path, format="wav")
            self.temp_files.append(processed_path)
            
            logger.info(f"Enhanced audio preprocessing complete. Duration reduced from "
                       f"{len(audio)/1000:.2f}s to {len(processed_audio)/1000:.2f}s")
            return processed_path
        except Exception as e:
            logger.warning(f"Enhanced audio preprocessing failed: {e}. Using original audio.")
            return audio_path
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using faster-whisper with enhanced settings.
        
        Args:
            audio_path (str): Path to the WAV audio file.
            
        Returns:
            str: Transcribed text.
        """
        logger.info("Transcribing audio with enhanced faster-whisper settings...")
        start_time = time.time()
        
        # Preprocess the audio for better quality
        processed_audio = self.preprocess_audio(audio_path)
        
        # Transcribe with optimized settings
        segments, info = self.whisper_model.transcribe(
            processed_audio,
            beam_size=5,           # Higher beam size for better accuracy
            language=None,         # Auto-detect language
            vad_filter=True,       # Voice activity detection
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Filter silence
                threshold=0.45                # Slightly more aggressive VAD
            ),
            condition_on_previous_text=True,  # Context continuity
            word_timestamps=True              # Get word-level timing
        )
        
        # Collect all segments
        transcribed_text = ""
        for segment in segments:
            # Add proper spacing based on context
            if transcribed_text and not transcribed_text.endswith(('.', '?', '!', '\n')):
                transcribed_text += " "
            transcribed_text += segment.text
        
        transcription_time = time.time() - start_time
        logger.info(f"Enhanced transcription completed in {transcription_time:.2f} seconds")
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Clean up memory
        if self.use_gpu:
            self._clear_memory()
            
        return transcribed_text.strip(), info.language
    
    def translate_text(self, text, target_lang="hi", source_lang=None):
        """
        Enhanced translation using a two-tier approach for optimal speed and accuracy.
        
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
            
            # Enhanced quality verification criteria
            suspicious_translation = False
            
            # 1. Check if result is significantly shorter than expected
            if len(translated) < len(text) * 0.5 and len(text) > 100:
                suspicious_translation = True
                logger.warning("Translation appears too short, might be low quality")
            
            # 2. Check for excessive repetition which indicates poor translation
            if len(text) > 50 and len(set(translated.split())) / len(translated.split()) < 0.4:
                suspicious_translation = True
                logger.warning("Translation contains excessive repetition, might be low quality")
            
            # 3. Check if translation contains original language words mixed in
            # (Simple heuristic, will vary by language)
            if len(text) > 50 and any(word in translated for word in text.split() if len(word) > 5):
                suspicious_translation = True
                logger.warning("Translation appears to contain untranslated words")
                
            if suspicious_translation:
                raise ValueError("Potentially low-quality translation, trying Gemini instead")
                
            translation_time = time.time() - start_time
            logger.info(f"Google Translate completed in {translation_time:.2f} seconds")
            return translated.strip()
            
        except Exception as e:
            logger.warning(f"Google Translate failed or produced low-quality result: {e}")
            logger.info("Trying Gemini for better accuracy...")
            
            try:
                gemini_start = time.time()
                
                # Create a more detailed prompt for Gemini with contextual guidance
                prompt = f"""Please translate the following text to {target_lang}. 
                Maintain the original style, tone, formality level, and format.
                Preserve paragraph breaks and ensure names and technical terms are handled correctly.
                
                Text to translate:
                ---
                {text}
                ---
                
                Translation (in {target_lang}):"""
                
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
        Translate long text by splitting it into paragraphs with enhanced context preservation.
        
        Args:
            text (str): Long text to translate.
            target_lang (str): Target language code.
            source_lang (str): Source language code. Default is None (auto-detect).
            
        Returns:
            str: Combined translated text.
        """
        logger.info("Using optimized large text translation...")
        
        # Step 1: Split by paragraphs (preserve structure)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Step 2: Further split extremely long paragraphs while preserving context
        split_paragraphs = []
        for para in paragraphs:
            if len(para) > 1000:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                # Group sentences into chunks with context overlap
                current = ""
                for sentence in sentences:
                    if len(current) + len(sentence) < 900:  # Slightly less than 1000 to allow context
                        current += sentence + " "
                    else:
                        if current:
                            split_paragraphs.append(current.strip())
                            # Keep the last sentence as context for next chunk
                            last_sentence = current.split(".")[-2] + "." if "." in current else ""
                            current = last_sentence + sentence + " "
                        else:
                            current = sentence + " "
                if current:
                    split_paragraphs.append(current.strip())
            else:
                split_paragraphs.append(para)
        
        # Step 3: Process in parallel batches for efficiency
        translated_parts = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.batch_size, len(split_paragraphs))) as executor:
            futures = {
                executor.submit(self.translate_text, part, target_lang, source_lang): i 
                for i, part in enumerate(split_paragraphs)
            }
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    translated_parts.append((idx, result))
                except Exception as e:
                    logger.error(f"Error translating part {idx}: {e}")
                    # Fall back to empty string to maintain structure
                    translated_parts.append((idx, ""))
        
        # Sort by original index and join with double newlines to preserve paragraph structure
        translated_parts.sort(key=lambda x: x[0])
        return "\n\n".join([part for _, part in translated_parts])
    
    def smart_split_text(self, text):
        """
        Enhanced text splitting that preserves semantic units and natural pauses.
        
        Args:
            text (str): Text to split.
            
        Returns:
            list: List of text chunks.
        """
        # This regex covers punctuation for multiple languages
        sentence_endings = r'[.!?।؛؟\n]'
        paragraph_breaks = r'\n\s*\n'
        
        # First, respect paragraph breaks
        paragraphs = re.split(paragraph_breaks, text)
        
        # Process each paragraph
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                # Small paragraph fits entirely
                chunks.append(paragraph)
                continue
                
            # Split while keeping the punctuation
            parts = re.split(f'({sentence_endings})', paragraph)
            
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
    
            current_chunk = ""
            
            for sentence in sentences:
                # If adding the sentence exceeds the chunk_size, start a new chunk
                if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                    else:
                        # If the sentence itself is longer than chunk_size, smart-split it
                        # Split on clause boundaries like commas, semicolons, etc.
                        clause_boundaries = r'[,;:-]'
                        clauses = re.split(f'({clause_boundaries})', sentence)
                        
                        if len(clauses) > 1:
                            # We have clauses we can split on
                            temp_chunk = ""
                            for i in range(0, len(clauses)-1, 2):
                                if i+1 < len(clauses):
                                    clause = clauses[i].strip() + clauses[i+1].strip()
                                    if len(temp_chunk) + len(clause) + 1 <= self.chunk_size:
                                        temp_chunk += clause + " "
                                    else:
                                        if temp_chunk:
                                            chunks.append(temp_chunk.strip())
                                        temp_chunk = clause + " "
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                        else:
                            # No clause boundaries, forced to hard-split by character
                            while len(sentence) > self.chunk_size:
                                # Try to split at a space within the last ~10% of chunk_size
                                split_point = self.chunk_size
                                while split_point > self.chunk_size * 0.9 and sentence[split_point] != ' ':
                                    split_point -= 1
                                    
                                # If no space found, do a hard split
                                if split_point <= self.chunk_size * 0.9:
                                    split_point = self.chunk_size
                                
                                chunks.append(sentence[:split_point].strip())
                                sentence = sentence[split_point:].strip()
                            
                            if sentence:
                                current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
                    
            if current_chunk:
                chunks.append(current_chunk.strip())
            
        logger.info(f"Enhanced text splitting produced {len(chunks)} semantically meaningful chunks")
        return chunks
    
    def compute_speaker_embedding(self, input_audio):
        """
        Compute and cache speaker embedding for reuse across TTS chunks.
        
        Args:
            input_audio (str): Path to the reference voice audio file.
            
        Returns:
            Any: Speaker embedding object that can be reused.
        """
        # Use cached embedding if available
        audio_hash = str(hash(input_audio))
        if audio_hash in self._speaker_embeddings:
            logger.info("Using cached speaker embedding")
            return self._speaker_embeddings[audio_hash]
        
        logger.info("Computing speaker embedding...")
        start_time = time.time()
        
        # Ensure TTS model is loaded
        tts_model = self.tts_model
        
        # Compute the embedding
        speaker_embedding = tts_model.synthesizer.tts_model.speaker_manager.compute_embedding(input_audio)
        
        # Cache the embedding
        self._speaker_embeddings[audio_hash] = speaker_embedding
        
        logger.info(f"Speaker embedding computed in {time.time() - start_time:.2f} seconds")
        return speaker_embedding
    
    def clone_voice_xtts_optimized(self, input_audio, translated_text, target_lang="hi"):
        """
        Optimized voice cloning using batched processing with pre-computed speaker embedding.
        
        Args:
            input_audio (str): Path to the reference voice audio file.
            translated_text (str): Translated text to synthesize.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            tuple: Tuple containing (list of audio part paths, combined audio path).
        """
        logger.info("Cloning voice with optimized batch processing...")
        text_chunks = self.smart_split_text(translated_text)
        
        # Pre-compute speaker embedding once
        speaker_embedding = self.compute_speaker_embedding(input_audio)
        
        # Process chunks in parallel batches for efficiency
        cloned_audio_paths = []
        temp_results = [None] * len(text_chunks)
        
        # Function for processing a single chunk
        def process_chunk(idx, chunk):
            try:
                logger.info(f"Synthesizing part {idx+1}/{len(text_chunks)}...")
                start_time = time.time()
                
                part_path = os.path.join(self.output_path, f"cloned_audio_part_{idx + 1}.wav")
                
                # Use pre-computed speaker embedding
                self.tts_model.tts_to_file(
                    text=chunk,
                    file_path=part_path,
                    speaker_wav=None,  # Don't recompute embedding
                    speaker_embedding=speaker_embedding,
                    language=target_lang,
                )
                
                synthesis_time = time.time() - start_time
                logger.info(f"Part {idx+1} synthesized in {synthesis_time:.2f} seconds")
                return idx, part_path
            except Exception as e:
                logger.error(f"Error processing chunk {idx+1}: {e}")
                return idx, None
        
        # Use ThreadPoolExecutor for parallel processing with controlled batch size
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.batch_size, len(text_chunks))) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(text_chunks)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, part_path = future.result()
                    if part_path:
                        temp_results[idx] = part_path
                except Exception as e:
                    logger.error(f"Error in TTS batch processing: {e}")
        
        # Filter out None values and keep the order
        cloned_audio_paths = [path for path in temp_results if path]
        
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
        Combine multiple audio files with enhanced audio transitions.
        
        Args:
            audio_paths (list): List of paths to audio files.
            output_path (str): Path to save the combined audio.
        """
        logger.info(f"Combining {len(audio_paths)} audio parts with smooth transitions...")
        combined = AudioSegment.empty()
        
        for i, path in enumerate(audio_paths):
            segment = AudioSegment.from_file(path)
            
            # Apply slight fade-in to first segment and fade-out to last segment
            if i == 0:
                segment = segment.fade_in(50)
            if i == len(audio_paths) - 1:
                segment = segment.fade_out(100)
                
            # Apply crossfade between segments for smoother transitions
            if combined.duration_seconds > 0:
                combined = combined.append(segment, crossfade=30)
            else:
                combined += segment
            
        # Normalize volume of final audio
        combined = combined.normalize()
        
        # Optimize dynamic range for better clarity
        # This compresses the audio slightly to make quieter parts more audible
        # while preventing louder parts from clipping
        threshold = -20.0  # dB
        ratio = 1.5
        attack = 5.0  # ms
        release = 50.0  # ms
        
        combined = combined.compress_dynamic_range(
            threshold=threshold,
            ratio=ratio, 
            attack=attack,
            release=release
        )
        
        combined.export(output_path, format="wav")
        logger.info(f"Enhanced combined audio saved to: {output_path}")
    
    def translate_and_clone(self, input_audio, target_lang="hi"):
        """
        Full pipeline: Transcribe, Translate, and Clone Voice with advanced optimizations.
        
        Args:
            input_audio (str): Path to the input audio file.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            dict: Results including original text, translated text, and audio paths.
        """
        logger.info(f"🚀 Starting enhanced translation and cloning to {target_lang}...")
        total_start_time = time.time()
        
        # Step 1: Convert audio to WAV
        input_wav = self.convert_to_wav(input_audio)
        logger.info(f"🎵 Converted audio to WAV: {input_wav}")
        
        # Step 2: Transcribe with whisper (using enhanced approach)
        original_text, detected_lang = self.transcribe_audio(input_wav)
        logger.info(f"📄 Original Text: {original_text[:100]}...")
        
        # Unload Whisper model to free memory
        self._whisper_model = None
        self._clear_memory()
        
        # Step 3: Translate (with enhanced quality checks)
        translated_text = self.translate_text(original_text, target_lang, source_lang=detected_lang)
        logger.info(f"🌐 Translated Text: {translated_text[:100]}...")
        
        # Step 4: Clone voice with batched approach
        audio_parts, combined_audio = self.clone_voice_xtts_optimized(input_wav, translated_text, target_lang)
        
        # Unload TTS model to free memory
        self._tts_model = None
        self._clear_memory()
        
        total_time = time.time() - total_start_time
        logger.info(f"✅ Done! Enhanced processing completed in {total_time:.2f} seconds")
        
        # Clean up temporary files if requested
        if self.cleanup_temp:
            self._cleanup()
        
        return {
            "original_text": original_text,
            "detected_language": detected_lang,
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
        self._speaker_embeddings = {}  # Clear cached embeddings
        self._clear_memory()


class StreamingAudioTranslator(EnhancedAudioTranslator):
    """
    Enhanced streaming version that implements pipeline processing with progress tracking.
    Each step starts as soon as data is available from the previous step.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_queue = {}
        self._lock = threading.Lock()
        self._progress_callbacks = {}
    
    def register_progress_callback(self, job_id, callback):
        """Register a callback function for progress updates."""
        with self._lock:
            self._progress_callbacks[job_id] = callback
    
    def update_progress(self, job_id, stage, progress, data=None):
        """Update progress for a job."""
        with self._lock:
            if job_id in self.results_queue:
                self.results_queue[job_id]["progress"] = {
                    "stage": stage,
                    "percentage": progress
                }
                
                # Call the callback if registered
                if job_id in self._progress_callbacks:
                    self._progress_callbacks[job_id](stage, progress, data)
    
    def _transcription_worker(self, job_id, input_audio, target_lang="hi"):
        """Worker thread for transcription."""
        try:
            # Update progress
            self.update_progress(job_id, "preprocessing", 0)
            
            # Convert to WAV
            input_wav = self.convert_to_wav(input_audio)
            self.update_progress(job_id, "preprocessing", 50)
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(input_wav)
            self.update_progress(job_id, "preprocessing", 100)
            
            # Update progress
            self.update_progress(job_id, "transcription", 0)
            
            # Transcribe in smaller segments for faster first results
            segments, info = self.whisper_model.transcribe(
                processed_audio,
                beam_size=5,
                language=None,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.45
                ),
                condition_on_previous_text=True,
                word_timestamps=True
            )
            
            # Process segments as they become available
            transcribed_text = ""
            segment_count = 0
            total_segments = 20  # Estimate, will be refined
            
            # Start sending segments as they become available
            for segment in segments:
                segment_count += 1
                transcribed_text += segment.text + " "
                
                # Update progress based on current segment count
                progress = min(95, (segment_count / max(total_segments, segment_count)) * 100)
                self.update_progress(job_id, "transcription", progress, 
                                   {"partial_text": transcribed_text.strip()})
            
            # Final transcription result
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["transcription"] = {
                        "text": transcribed_text.strip(),
                        "language": info.language
                    }
            
            self.update_progress(job_id, "transcription", 100, 
                               {"text": transcribed_text.strip(), "language": info.language})
            
            # Start the translation worker
            translation_thread = threading.Thread(
                target=self._translation_worker, 
                args=(job_id, transcribed_text.strip(), info.language, target_lang)
            )
            translation_thread.start()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["error"] = f"Transcription failed: {e}"
            self.update_progress(job_id, "error", 100, {"error": str(e)})
    
    def _translation_worker(self, job_id, text, source_lang, target_lang):
        """Worker thread for translation."""
        try:
            # Skip translation if source and target languages are the same
            if source_lang == target_lang:
                logger.info(f"Source and target languages are the same ({source_lang}), skipping translation")
                translated_text = text
                self.update_progress(job_id, "translation", 100, {"text": translated_text})
                
                # Start the TTS worker immediately
                tts_thread = threading.Thread(
                    target=self._tts_worker, 
                    args=(job_id, text, target_lang)
                )
                tts_thread.start()
                return
            
            # Update progress
            self.update_progress(job_id, "translation", 0)
            
            # For longer texts, split and translate in chunks with progress updates
            if len(text) > 1500:
                # Split text into paragraphs
                paragraphs = re.split(r'\n\s*\n', text)
                
                # Initialize progress tracking
                total_chars = sum(len(p) for p in paragraphs)
                processed_chars = 0
                translated_paragraphs = []
                
                # Process each paragraph
                for i, paragraph in enumerate(paragraphs):
                    # Translate the paragraph
                    translated = self.translate_text(paragraph, target_lang, source_lang)
                    translated_paragraphs.append(translated)
                    
                    # Update progress
                    processed_chars += len(paragraph)
                    progress = min(95, (processed_chars / total_chars) * 100)
                    
                    # Join processed paragraphs for partial results
                    partial_result = "\n\n".join(translated_paragraphs)
                    self.update_progress(job_id, "translation", progress, {"partial_text": partial_result})
                
                # Join all translated paragraphs
                translated_text = "\n\n".join(translated_paragraphs)
            else:
                # For shorter texts, translate all at once
                translated_text = self.translate_text(text, target_lang, source_lang)
                self.update_progress(job_id, "translation", 90, {"partial_text": translated_text})
            
            # Store the result
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["translation"] = {
                        "text": translated_text
                    }
            
            self.update_progress(job_id, "translation", 100, {"text": translated_text})
            
            # Start the TTS worker
            tts_thread = threading.Thread(
                target=self._tts_worker, 
                args=(job_id, translated_text, target_lang)
            )
            tts_thread.start()
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["error"] = f"Translation failed: {e}"
            self.update_progress(job_id, "error", 100, {"error": str(e)})
    
    def _tts_worker(self, job_id, text, target_lang):
        """Worker thread for text-to-speech processing."""
        try:
            # Update progress
            self.update_progress(job_id, "voice_cloning", 0)
            
            # Get the input audio path
            with self._lock:
                if job_id not in self.results_queue:
                    logger.error(f"Job ID {job_id} not found in results queue")
                    return
                input_audio = self.results_queue[job_id]["input_audio"]
            
            # Split text into manageable chunks for TTS
            chunks = self.smart_split_text(text)
            
            # Pre-compute speaker embedding
            speaker_embedding = self.compute_speaker_embedding(input_audio)
            
            # Initialize progress tracking
            total_chunks = len(chunks)
            processed_chunks = 0
            audio_parts = []
            
            # Process chunks in parallel batches for efficiency
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.batch_size, total_chunks)) as executor:
                # Submit all chunks for processing
                futures = {}
                for i, chunk in enumerate(chunks):
                    future = executor.submit(
                        self._process_tts_chunk, 
                        chunk, 
                        i, 
                        speaker_embedding, 
                        target_lang
                    )
                    futures[future] = i
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    idx, part_path = future.result()
                    
                    if part_path:
                        # Keep track of processed parts
                        audio_parts.append((idx, part_path))
                        
                        # Update progress
                        processed_chunks += 1
                        progress = min(95, (processed_chunks / total_chunks) * 100)
                        
                        # Report partial results
                        current_parts = sorted(audio_parts, key=lambda x: x[0])
                        partial_paths = [path for _, path in current_parts]
                        
                        self.update_progress(
                            job_id, 
                            "voice_cloning", 
                            progress, 
                            {"processed_chunks": processed_chunks, "total_chunks": total_chunks}
                        )
            
            # Sort parts by index and extract just the paths
            sorted_parts = sorted(audio_parts, key=lambda x: x[0])
            part_paths = [path for _, path in sorted_parts]
            
            # Combine all audio parts
            combined_path = os.path.join(self.output_path, f"{job_id}_combined.wav")
            self._combine_audio_files(part_paths, combined_path)
            
            # Store the result
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["voice_cloning"] = {
                        "audio_parts": part_paths,
                        "combined_audio": combined_path
                    }
            
            self.update_progress(job_id, "voice_cloning", 100, {
                "audio_parts": part_paths,
                "combined_audio": combined_path
            })
            
            # Mark job as completed
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["status"] = "completed"
                    self.results_queue[job_id]["completion_time"] = time.time()
            
            self.update_progress(job_id, "completed", 100, {
                "combined_audio": combined_path
            })
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            with self._lock:
                if job_id in self.results_queue:
                    self.results_queue[job_id]["error"] = f"Voice cloning failed: {e}"
            self.update_progress(job_id, "error", 100, {"error": str(e)})
    
    def _process_tts_chunk(self, chunk, idx, speaker_embedding, target_lang):
        """Process a single TTS chunk."""
        try:
            logger.info(f"Synthesizing part {idx+1} with XTTS...")
            start_time = time.time()
            
            part_path = os.path.join(self.output_path, f"part_{idx+1}.wav")
            
            # Use pre-computed speaker embedding
            self.tts_model.tts_to_file(
                text=chunk,
                file_path=part_path,
                speaker_wav=None,  # Don't recompute embedding
                speaker_embedding=speaker_embedding,
                language=target_lang,
            )
            
            synthesis_time = time.time() - start_time
            logger.info(f"Part {idx+1} synthesized in {synthesis_time:.2f} seconds")
            return idx, part_path
        except Exception as e:
            logger.error(f"Error processing TTS chunk {idx+1}: {e}")
            return idx, None
    
    def process_audio_streaming(self, input_audio, target_lang="hi"):
        """
        Process audio with streaming updates.
        
        Args:
            input_audio (str): Path to the input audio file.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            str: Job ID for tracking progress.
        """
        # Create a unique job ID
        job_id = f"job_{int(time.time())}_{hash(input_audio) % 10000}"
        
        # Initialize job in the queue
        with self._lock:
            self.results_queue[job_id] = {
                "status": "started",
                "start_time": time.time(),
                "input_audio": input_audio,
                "target_language": target_lang,
                "progress": {
                    "stage": "initializing",
                    "percentage": 0
                }
            }
        
        # Start the transcription process in a separate thread
        transcription_thread = threading.Thread(
            target=self._transcription_worker, 
            args=(job_id, input_audio, target_lang)
        )
        transcription_thread.start()
        
        return job_id
    
    def get_job_status(self, job_id):
        """
        Get the current status of a job.
        
        Args:
            job_id (str): Job ID to check.
            
        Returns:
            dict: Current job status and progress information.
        """
        with self._lock:
            if job_id not in self.results_queue:
                return {"error": "Job not found"}
            return self.results_queue[job_id]
    
    def cleanup_job(self, job_id):
        """
        Clean up job resources.
        
        Args:
            job_id (str): Job ID to clean up.
        """
        with self._lock:
            if job_id in self.results_queue:
                # Get any audio paths to clean up
                job_data = self.results_queue[job_id]
                if "voice_cloning" in job_data:
                    voice_cloning = job_data["voice_cloning"]
                    if "audio_parts" in voice_cloning:
                        for part in voice_cloning["audio_parts"]:
                            try:
                                os.remove(part)
                            except Exception as e:
                                logger.warning(f"Failed to remove audio part {part}: {e}")
                
                # Remove the job from the queue
                del self.results_queue[job_id]


def main():
    """Main entry point function for command-line usage."""
    parser = argparse.ArgumentParser(description="Enhanced Audio Translator")
    parser.add_argument("input_audio", help="Path to the input audio file")
    parser.add_argument("--target-lang", "-t", default="hi", help="Target language code (default: hi)")
    parser.add_argument("--output-path", "-o", help="Output directory path")
    parser.add_argument("--whisper-model", "-w", default="medium", help="Whisper model size (default: medium)")
    parser.add_argument("--chunk-size", "-c", type=int, default=250, help="TTS chunk size (default: 250)")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Processing batch size (default: 4)")
    parser.add_argument("--streaming", "-s", action="store_true", help="Use streaming mode")
    parser.add_argument("--gpu", "-g", action="store_true", help="Use GPU if available (default: True)")
    parser.add_argument("--no-cleanup", "-n", action="store_true", help="Don't clean up temporary files")
    
    args = parser.parse_args()
    
    try:
        # Create the translator instance
        if args.streaming:
            translator = StreamingAudioTranslator(
                output_path=args.output_path,
                chunk_size=args.chunk_size,
                cleanup_temp=not args.no_cleanup,
                whisper_model_size=args.whisper_model,
                use_gpu=args.gpu,
                batch_size=args.batch_size
            )
            
            # Handle streaming mode with progress updates
            def progress_callback(stage, percentage, data):
                if stage == "completed":
                    print(f"\n✅ Processing completed! Output saved to: {data.get('combined_audio')}")
                elif stage == "error":
                    print(f"\n❌ Error: {data.get('error')}")
                else:
                    print(f"\r{stage.capitalize()}: {percentage:.1f}%", end="")
            
            # Start processing
            job_id = translator.process_audio_streaming(args.input_audio, args.target_lang)
            translator.register_progress_callback(job_id, progress_callback)
            
            # Wait for completion (in a real app, this would be handled by the frontend)
            print(f"Started job {job_id}")
            while True:
                status = translator.get_job_status(job_id)
                if status.get("status") == "completed" or "error" in status:
                    break
                time.sleep(1)
                
            translator.unload_models()
            translator.cleanup_job(job_id)
            
        else:
            # Use the standard synchronous approach
            translator = EnhancedAudioTranslator(
                output_path=args.output_path,
                chunk_size=args.chunk_size,
                cleanup_temp=not args.no_cleanup,
                whisper_model_size=args.whisper_model,
                use_gpu=args.gpu,
                batch_size=args.batch_size
            )
            
            result = translator.translate_and_clone(args.input_audio, args.target_lang)
            
            print("\n✅ Processing completed!")
            print(f"Original text: {result['original_text'][:100]}...")
            print(f"Detected language: {result['detected_language']}")
            print(f"Translated text: {result['translated_text'][:100]}...")
            print(f"Combined audio saved to: {result['combined_audio']}")
            print(f"Total processing time: {result['processing_time_seconds']:.2f} seconds")
            
            translator.unload_models()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())