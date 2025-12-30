"""
Real-time speaker identification processor with threading support.
"""

import collections
import numpy as np
import torch
import threading
import queue
import time
from functools import lru_cache

from src.utils.vad import VADBuffer, VoiceActivityDetector
from src.audio.preprocessing import extract_mel_spectrogram
from src.utils.logging_config import get_logger
from src.utils import config_loader as config

logger = get_logger("realtime")


class ThreadedAudioProcessor:
    """
    Threaded audio processor for real-time identification.
    Separates audio capture from inference for better latency.
    """
    def __init__(self, identifier, sample_rate=16000, buffer_duration=None, vad_confidence=None):
        self.identifier = identifier
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration or config.get('inference.rolling_window_size', 2.0)
        self.model_sr = config.sample_rate()  # Model sample rate from config
        
        # VAD with pre-roll buffering
        self.vad_buffer = VADBuffer(sample_rate, preroll_duration=0.5, confidence_threshold=vad_confidence)
        
        # Threading
        self.audio_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        self.inference_thread = None
        
        # Stats
        self.last_result = None
        self.speech_probability = 0.0
        
        # LRU cache for embeddings
        self._embedding_cache = {}
        
    def start(self):
        """Start the inference thread."""
        if self.running:
            return
            
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        logger.info("Inference thread started.")
        
    def stop(self):
        """Stop the inference thread."""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        self.vad_buffer.reset()
        logger.info("Inference thread stopped.")
        
    def push_audio(self, audio_chunk: np.ndarray):
        """
        Push audio chunk to processing queue.
        audio_chunk: int16 numpy array
        """
        try:
            self.audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            pass  # Drop oldest data
            
    def get_result(self):
        """Get latest inference result (non-blocking)."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return self.last_result
            
    def _inference_loop(self):
        """Background inference thread."""
        accumulated_audio = []
        accumulated_samples = 0
        target_samples = int(self.sample_rate * self.buffer_duration)
        
        while self.running:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Convert to float for VAD
                if chunk.dtype == np.int16:
                    chunk_float = chunk.astype(np.float32) / 32768.0
                else:
                    chunk_float = chunk.astype(np.float32)
                
                # Process through VAD buffer
                speech_segment, prob = self.vad_buffer.process_chunk(chunk_float)
                self.speech_probability = prob
                
                if speech_segment is not None:
                    # VAD detected complete speech segment
                    result = self._process_segment(speech_segment)
                    if result:
                        self.last_result = result
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            pass
                else:
                    # No complete segment yet - use rolling buffer approach
                    accumulated_audio.append(chunk_float)
                    accumulated_samples += len(chunk_float)
                    
                    if accumulated_samples >= target_samples:
                        # Process accumulated audio
                        full_audio = np.concatenate(accumulated_audio)[-target_samples:]
                        
                        # Only process if there's speech
                        if prob >= 0.3:
                            result = self._process_segment(full_audio)
                            if result:
                                result['is_rolling'] = True
                                self.last_result = result
                                try:
                                    self.result_queue.put_nowait(result)
                                except queue.Full:
                                    pass
                                    
                        # Trim buffer
                        accumulated_audio = [full_audio[-target_samples//2:]]
                        accumulated_samples = len(accumulated_audio[0])
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference error: {e}")
                time.sleep(0.1)
                
    def _process_segment(self, audio_segment: np.ndarray) -> dict:
        """Process a speech segment and return identification result."""
        try:
            # Resample to model sample rate
            if self.sample_rate != self.model_sr:
                import librosa
                resampled = librosa.resample(audio_segment, orig_sr=self.sample_rate, target_sr=self.model_sr)
            else:
                resampled = audio_segment
                
            # Compute embedding
            embedding = self._compute_embedding(resampled)
            if embedding is None:
                return None
                
            # Match against enrolled speakers
            best_score = -1.0
            best_speaker = "Unknown"
            all_scores = {}
            
            for name, ref_emb in self.identifier.speakers.items():
                score = torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0), 
                    ref_emb.unsqueeze(0)
                ).item()
                all_scores[name] = score
                if score > best_score:
                    best_score = score
                    best_speaker = name
                    
            # Check threshold from config
            threshold = config.threshold()
            if best_score < threshold:
                best_speaker = "Unknown"
                
            return {
                "speaker": best_speaker,
                "confidence": best_score,
                "all_scores": all_scores,
                "speech_prob": self.speech_probability,
                "is_rolling": False
            }
            
        except Exception as e:
            logger.error(f"Segment processing error: {e}")
            return None
            
    def _compute_embedding(self, audio_array: np.ndarray) -> torch.Tensor:
        """Compute embedding from numpy audio array (8kHz float32)."""
        try:
            # Ensure correct length (1 sec)
            target_len = 8000
            if len(audio_array) < target_len:
                audio_array = np.pad(audio_array, (0, target_len - len(audio_array)))
            else:
                audio_array = audio_array[:target_len]
                
            # Extract spectrogram
            spec = extract_mel_spectrogram(audio_array, sr=8000)
            
            # To tensor
            t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.identifier.device)
            
            with torch.no_grad():
                emb = self.identifier.model(t)
                return torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0)
                
        except Exception as e:
            logger.error(f"Embedding computation error: {e}")
            return None


class RealTimeProcessor:
    """
    Legacy-compatible wrapper around ThreadedAudioProcessor.
    """
    def __init__(self, identifier, sample_rate=16000, buffer_duration=2.0):
        self.identifier = identifier
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.audio_buffer = collections.deque(maxlen=self.buffer_size)
        self.vad = VoiceActivityDetector(sample_rate)
        self.model_sr = 8000
    
    def process_chunk(self, audio_chunk_int16: np.ndarray):
        """
        Legacy interface for backward compatibility.
        """
        # Update buffer
        self.audio_buffer.extend(audio_chunk_int16)
        
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        full_buffer = np.array(self.audio_buffer, dtype=np.int16)
        
        # VAD check
        float_audio = full_buffer.astype(np.float32) / 32768.0
        speech_prob = self.vad.get_speech_prob(float_audio[-1600:])  # Last 100ms
        
        if speech_prob < 0.3:
            return {"speaker": "Silence", "confidence": 0.0}
            
        # Resample and inference
        import librosa
        if self.sample_rate != self.model_sr:
            resampled = librosa.resample(float_audio, orig_sr=self.sample_rate, target_sr=self.model_sr)
        else:
            resampled = float_audio
            
        embedding = self._compute_embedding_from_array(resampled)
        if embedding is None:
            return None
             
        # Match
        best_score = -1.0
        best_speaker = "Unknown"
        
        for name, ref_emb in self.identifier.speakers.items():
            score = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), 
                ref_emb.unsqueeze(0)
            ).item()
            if score > best_score:
                best_score = score
                best_speaker = name
                
        return {"speaker": best_speaker, "confidence": best_score}

    def _compute_embedding_from_array(self, audio_array):
        try:
            if len(audio_array) < 8000:
                pad_len = 8000 - len(audio_array)
                audio_array = np.pad(audio_array, (0, pad_len))
            else:
                audio_array = audio_array[:8000]
                 
            spec = extract_mel_spectrogram(audio_array, sr=8000)
            t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.identifier.device)
            
            with torch.no_grad():
                emb = self.identifier.model(t)
                return torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0)
        except Exception as e:
            return None
