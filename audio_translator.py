import os
import whisper
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
import re
import tempfile
import logging
import time
import argparse
from pathlib import Path
import concurrent.futures
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Allow XTTS config classes to be safely unpickled
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


# Standalone function for parallel processing
def process_tts_chunk_standalone(chunk_data):
    """
    Standalone function to process a single TTS chunk (for parallel processing).
    
    Args:
        chunk_data (tuple): Tuple containing (chunk_index, chunk_text, input_audio, target_lang, part_path, device, optimize_memory).
        
    Returns:
        tuple: Tuple containing (chunk_index, part_path).
    """
    chunk_index, chunk_text, input_audio, target_lang, part_path, device, optimize_memory = chunk_data
    
    try:
        logger.info(f"Synthesizing part {chunk_index+1}...")
        start_time = time.time()
        
        # Create a local TTS model instance
        local_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if device == "cuda":
            # Use a specific GPU if available
            local_tts = local_tts.to("cuda")
        
        local_tts.tts_to_file(
            text=chunk_text,
            speaker_wav=input_audio,
            language=target_lang,
            file_path=part_path,
        )
        
        synthesis_time = time.time() - start_time
        logger.info(f"Part {chunk_index+1} synthesized in {synthesis_time:.2f} seconds")
        
        # Clear memory if optimizing
        if optimize_memory and device == "cuda":
            del local_tts
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                
        return chunk_index, part_path
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index+1}: {e}")
        return chunk_index, None


class AudioTranslator:
    def __init__(self, output_path=None, chunk_size=250, cleanup_temp=True, 
                 parallel_processing=True, max_workers=None, optimize_memory=True,
                 whisper_model_size="base"):
        """
        Initialize the AudioTranslator with configuration options.
        
        Args:
            output_path (str): Directory to save output files. Default is "output_audio".
            chunk_size (int): Maximum character length for TTS chunks. Default is 250.
            cleanup_temp (bool): Whether to clean up temporary files. Default is True.
            parallel_processing (bool): Whether to use parallel processing for TTS. Default is True.
            max_workers (int): Maximum number of workers for parallel processing. Default is None (auto).
            optimize_memory (bool): Whether to optimize memory usage. Default is True.
            whisper_model_size (str): Size of the Whisper model. Default is "base".
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chunk_size = chunk_size
        self.cleanup_temp = cleanup_temp
        self.temp_files = []
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        self.optimize_memory = optimize_memory
        self.whisper_model_size = whisper_model_size
        
        # Initialize models lazily (they'll be loaded when needed)
        self._whisper_model = None
        self._tts_model = None
        self._gemini_model = None
        
        logger.info(f"Initialized AudioTranslator (device: {self.device}, chunk size: {self.chunk_size}, "
                   f"parallel: {parallel_processing}, whisper model: {whisper_model_size})")
    
    @property
    def whisper_model(self):
        """Lazy loading of the Whisper model."""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper {self.whisper_model_size} model...")
            start_time = time.time()
            self._whisper_model = whisper.load_model(self.whisper_model_size)
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
        if self.optimize_memory and torch.cuda.is_available():
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
            
        Raises:
            ValueError: If the audio file cannot be converted.
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
    
    def split_audio_for_transcription(self, audio_path, segment_length_ms=60000):
        """
        Split long audio into shorter segments for faster transcription.
        
        Args:
            audio_path (str): Path to the WAV audio file.
            segment_length_ms (int): Length of each segment in milliseconds. Default is 60000 (1 minute).
            
        Returns:
            list: List of paths to temporary segment files.
        """
        try:
            full_audio = AudioSegment.from_file(audio_path)
            total_length_ms = len(full_audio)
            
            if total_length_ms <= segment_length_ms:
                # No need to split
                return [audio_path]
            
            logger.info(f"Splitting {total_length_ms/1000:.2f}s audio into {segment_length_ms/1000:.0f}s segments")
            
            segment_paths = []
            for i in range(0, total_length_ms, segment_length_ms):
                end = min(i + segment_length_ms, total_length_ms)
                segment = full_audio[i:end]
                
                fd, segment_path = tempfile.mkstemp(suffix=f"_seg{i//segment_length_ms}.wav")
                os.close(fd)
                segment.export(segment_path, format="wav")
                segment_paths.append(segment_path)
                self.temp_files.append(segment_path)
            
            logger.info(f"Split audio into {len(segment_paths)} segments")
            return segment_paths
        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [audio_path]  # Fall back to the original file
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path (str): Path to the WAV audio file.
            
        Returns:
            str: Transcribed text.
        """
        logger.info("Transcribing audio...")
        start_time = time.time()
        
        # Split long audio for faster processing
        audio_segments = self.split_audio_for_transcription(audio_path)
        
        if len(audio_segments) == 1:
            # Single segment
            result = self.whisper_model.transcribe(audio_path)
            transcribed_text = result["text"]
        else:
            # Multiple segments - transcribe each and combine
            transcribed_segments = []
            
            for i, segment_path in enumerate(audio_segments):
                logger.info(f"Transcribing segment {i+1}/{len(audio_segments)}...")
                result = self.whisper_model.transcribe(segment_path)
                transcribed_segments.append(result["text"])
                
                # Clear memory after each segment if requested
                if self.optimize_memory:
                    self._clear_memory()
            
            transcribed_text = " ".join(transcribed_segments)
        
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        return transcribed_text
    
    def translate_text(self, text, target_lang="hi"):
        """
        Translate text using Gemini (fallback to Google Translate if needed).
        
        Args:
            text (str): Text to translate.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            str: Translated text.
        """
        logger.info(f"Translating to '{target_lang}' using Gemini...")
        start_time = time.time()
        
        # Split very long text for better translation quality
        if len(text) > 5000:
            return self._translate_long_text(text, target_lang)
        
        try:
            prompt = f"Translate the following text to {target_lang}:\n{text}"
            response = self.gemini_model.generate_content(prompt)
            translated = response.text.strip()
            if not translated:
                raise ValueError("Empty translation from Gemini.")
            
            translation_time = time.time() - start_time
            logger.info(f"Gemini translation completed in {translation_time:.2f} seconds")
            return translated
        except Exception as e:
            logger.warning(f"Gemini translation failed: {e}")
            logger.info("Falling back to Google Translate...")
            
            try:
                fallback_start = time.time()
                translator = GoogleTranslator(source="auto", target=target_lang)
                translated = translator.translate(text)
                
                fallback_time = time.time() - fallback_start
                logger.info(f"Google Translate fallback completed in {fallback_time:.2f} seconds")
                return translated.strip()
            except Exception as e2:
                logger.error(f"Google Translate also failed: {e2}")
                return ""
    
    def _translate_long_text(self, text, target_lang):
        """
        Translate long text by splitting it into paragraphs.
        
        Args:
            text (str): Long text to translate.
            target_lang (str): Target language code.
            
        Returns:
            str: Combined translated text.
        """
        logger.info("Splitting long text for translation")
        
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
            translated = self.translate_text(part, target_lang)
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
        # This regex covers punctuation for English, Hindi (।), Arabic/Urdu (؟, ؛), and common ones
        sentence_endings = r'[.!?।؛؟\n]'
        # Split while keeping the punctuation as separate tokens
        parts = re.split(f'({sentence_endings})', text)
        # Combine each sentence with its punctuation
        sentences = []
        for i in range(0, len(parts)-1, 2):
            if i+1 < len(parts):
                sentence = parts[i].strip() + parts[i+1].strip()
                if sentence:
                    sentences.append(sentence)
        # In case there's any trailing text without punctuation
        if len(parts) % 2 != 0 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # If adding the sentence exceeds the chunk_size, then start a new chunk.
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    # If the sentence itself is longer than chunk_size, hard-split it.
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
    
    def process_tts_chunk(self, chunk_data):
        """
        Process a single TTS chunk (for sequential processing).
        
        Args:
            chunk_data (tuple): Tuple containing (chunk_index, chunk_text, input_audio, target_lang, part_path).
            
        Returns:
            tuple: Tuple containing (chunk_index, part_path).
        """
        chunk_index, chunk_text, input_audio, target_lang, part_path = chunk_data
        
        try:
            logger.info(f"Synthesizing part {chunk_index+1}...")
            start_time = time.time()
            
            self.tts_model.tts_to_file(
                text=chunk_text,
                speaker_wav=input_audio,
                language=target_lang,
                file_path=part_path,
            )
            
            synthesis_time = time.time() - start_time
            logger.info(f"Part {chunk_index+1} synthesized in {synthesis_time:.2f} seconds")
            
            # Clear memory if optimizing
            if self.optimize_memory and self.device == "cuda":
                self._clear_memory()
                
            return chunk_index, part_path
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index+1}: {e}")
            return chunk_index, None
    
    def clone_voice_xtts(self, input_audio, translated_text, target_lang="hi"):
        """
        Clone the voice using XTTS in chunks with optional parallel processing.
        
        Args:
            input_audio (str): Path to the reference voice audio file.
            translated_text (str): Translated text to synthesize.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            list: List of paths to cloned audio part files.
            str: Path to the combined audio file.
        """
        logger.info(f"Cloning voice with XTTS (parallel: {self.parallel_processing})...")
        text_chunks = self.smart_split_text(translated_text)
        
        # Prepare chunk data for processing
        part_paths = [None] * len(text_chunks)  # Preallocate to maintain order
        
        # Process chunks
        if self.parallel_processing and len(text_chunks) > 1:
            # Prepare data for parallel processing
            chunk_data = []
            for i, chunk in enumerate(text_chunks):
                part_path = os.path.join(self.output_path, f"cloned_audio_part_{i + 1}.wav")
                # Include device and optimize_memory in the chunk data for the standalone function
                chunk_data.append((i, chunk, input_audio, target_lang, part_path, self.device, self.optimize_memory))
            
            # Use parallel processing for multiple chunks
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Use the standalone function instead of the class method
                for chunk_index, part_path in executor.map(process_tts_chunk_standalone, chunk_data):
                    if part_path:
                        part_paths[chunk_index] = part_path
        else:
            # Process sequentially using the class method
            for i, chunk in enumerate(text_chunks):
                part_path = os.path.join(self.output_path, f"cloned_audio_part_{i + 1}.wav")
                chunk_index, part_path = self.process_tts_chunk((i, chunk, input_audio, target_lang, part_path))
                if part_path:
                    part_paths[i] = part_path
        
        # Filter out any None values (failed chunks)
        cloned_audio_paths = [path for path in part_paths if path]
        
        # Combine all audio parts
        combined_path = os.path.join(self.output_path, "translated_audio_combined.wav")
        if len(cloned_audio_paths) > 1:
            self._combine_audio_files(cloned_audio_paths, combined_path)
        elif cloned_audio_paths:
            combined_path = cloned_audio_paths[0]
        else:
            logger.error("No audio parts were successfully generated")
            combined_path = None
            
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
            
        combined.export(output_path, format="wav")
        logger.info(f"Combined audio saved to: {output_path}")
    
    def translate_and_clone(self, input_audio, target_lang="hi"):
        """
        Full pipeline: Transcribe, Translate, and Clone Voice.
        
        Args:
            input_audio (str): Path to the input audio file.
            target_lang (str): Target language code. Default is "hi" (Hindi).
            
        Returns:
            dict: Results including original text, translated text, and audio paths.
        """
        logger.info(f"🚀 Starting translation and cloning to {target_lang}...")
        total_start_time = time.time()
        
        # Step 1: Convert audio to WAV
        input_wav = self.convert_to_wav(input_audio)
        logger.info(f"🎵 Converted audio to WAV: {input_wav}")
        
        # Step 2: Transcribe
        original_text = self.transcribe_audio(input_wav)
        logger.info(f"📄 Original Text: {original_text[:100]}...")
        
        # Clear memory after transcription
        if self.optimize_memory:
            self._clear_memory()
        
        # Step 3: Translate
        translated_text = self.translate_text(original_text, target_lang)
        logger.info(f"🌐 Translated Text: {translated_text[:100]}...")
        
        # Step 4: Clone voice
        audio_parts, combined_audio = self.clone_voice_xtts(input_wav, translated_text, target_lang)
        
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

def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description="Audio Translation Pipeline")
    parser.add_argument("input_audio", help="Path to the input audio file")
    parser.add_argument("--target-lang", "-t", default="hi", help="Target language code (default: hi)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: output_audio)")
    parser.add_argument("--chunk-size", "-c", type=int, default=250, 
                        help="Maximum character length for TTS chunks (default: 250)")
    parser.add_argument("--whisper-model", "-w", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--max-workers", type=int, help="Maximum number of workers for parallel processing")
    parser.add_argument("--no-memory-optimization", action="store_true", help="Disable memory optimization")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    translator = AudioTranslator(
        output_path=args.output_dir,
        chunk_size=args.chunk_size,
        cleanup_temp=not args.keep_temp,
        parallel_processing=not args.no_parallel,
        max_workers=args.max_workers,
        optimize_memory=not args.no_memory_optimization,
        whisper_model_size=args.whisper_model
    )
    
    try:
        results = translator.translate_and_clone(args.input_audio, args.target_lang)
        
        print("\n--- Results ---")
        print(f"Original text: {results['original_text'][:100]}...")
        print(f"Translated text: {results['translated_text'][:100]}...")
        print(f"Combined audio: {results['combined_audio']}")
        print(f"Processing time: {results['processing_time_seconds']:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())