"""
Voice Activity Detection (VAD) Module

Implements Silero VAD (primary) with fallback to WebRTC and energy-based VAD.
Includes pre-roll buffering to capture audio before speech starts.
"""

import numpy as np
import logging
import collections
import torch

# --- Silero VAD (Primary) ---
_silero_model = None
_silero_utils = None

def get_silero_vad():
    """Lazy load Silero VAD model."""
    global _silero_model, _silero_utils
    if _silero_model is None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            _silero_model = model
            _silero_utils = utils
            logging.info("Silero VAD loaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to load Silero VAD: {e}. Falling back to alternatives.")
            _silero_model = False  # Mark as failed
    return _silero_model, _silero_utils

# --- WebRTC VAD (Fallback) ---
try:
    import webrtcvad
except ImportError:
    webrtcvad = None


class SileroVAD:
    """
    Silero-based Voice Activity Detector with confidence thresholding.
    """
    def __init__(self, sample_rate=16000, confidence_threshold=0.5):
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.model, self.utils = get_silero_vad()
        self.available = self.model is not None and self.model is not False
        
        if self.available:
            # Reset model state
            self.model.reset_states()
            
    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        Returns probability that the chunk contains speech.
        audio_chunk: numpy array of float32 samples
        """
        if not self.available:
            return 0.5  # Neutral fallback
            
        try:
            # Silero expects torch tensor
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_chunk)) > 1.0:
                audio_chunk = audio_chunk / 32768.0
                
            tensor = torch.from_numpy(audio_chunk)
            prob = self.model(tensor, self.sample_rate).item()
            return prob
        except Exception as e:
            logging.warning(f"Silero VAD error: {e}")
            return 0.5
            
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Returns True if chunk is likely speech."""
        return self.get_speech_prob(audio_chunk) >= self.confidence_threshold


class VoiceActivityDetector:
    """
    Multi-backend VAD with automatic fallback.
    Priority: Silero > WebRTC > Energy-based
    """
    def __init__(self, sample_rate=16000, confidence_threshold=0.5, aggressiveness=3):
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        
        # Try Silero first
        self.silero = SileroVAD(sample_rate, confidence_threshold)
        
        # WebRTC fallback
        self.webrtc_vad = None
        if not self.silero.available and webrtcvad:
            try:
                self.webrtc_vad = webrtcvad.Vad(aggressiveness)
                logging.info("Using WebRTC VAD as fallback.")
            except Exception as e:
                logging.warning(f"WebRTC VAD init failed: {e}")

    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability (0.0 - 1.0)."""
        if self.silero.available:
            return self.silero.get_speech_prob(audio_chunk)
        else:
            # WebRTC/Energy just returns 0 or 1
            return 1.0 if self.is_speech_bytes(audio_chunk.tobytes()) else 0.0
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if numpy array contains speech."""
        if self.silero.available:
            return self.silero.is_speech(audio_chunk)
        else:
            return self.is_speech_bytes(audio_chunk.tobytes())
            
    def is_speech_bytes(self, frame_bytes: bytes) -> bool:
        """Check if raw bytes contain speech (for WebRTC/Energy)."""
        if self.webrtc_vad:
            try:
                return self.webrtc_vad.is_speech(frame_bytes, self.sample_rate)
            except Exception:
                pass
        # Energy fallback
        return self._energy_vad(frame_bytes)

    def _energy_vad(self, frame_bytes: bytes, threshold=0.01) -> bool:
        """Simple RMS-based VAD."""
        audio_data = np.frombuffer(frame_bytes, dtype=np.int16)
        if len(audio_data) == 0:
            return False
        rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        return rms > (32768 * threshold)


class VADBuffer:
    """
    Smart buffer that keeps pre-roll audio and detects speech segments.
    """
    def __init__(self, sample_rate=16000, preroll_duration=0.5, confidence_threshold=0.5):
        self.sample_rate = sample_rate
        self.preroll_samples = int(preroll_duration * sample_rate)
        self.vad = VoiceActivityDetector(sample_rate, confidence_threshold)
        
        # Circular buffer for pre-roll
        self.preroll_buffer = collections.deque(maxlen=self.preroll_samples)
        
        # Speech state machine
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.silence_threshold = 10  # Number of non-speech frames to end segment
        
    def process_chunk(self, audio_chunk: np.ndarray):
        """
        Process audio chunk and return speech segment if complete.
        Returns: (speech_segment, speech_prob) or (None, prob)
        """
        prob = self.vad.get_speech_prob(audio_chunk)
        is_speech = prob >= self.vad.confidence_threshold
        
        if is_speech:
            if not self.is_speaking:
                # Speech started - include pre-roll
                self.is_speaking = True
                self.speech_buffer = list(self.preroll_buffer)
                self.speech_buffer.append(audio_chunk)
            else:
                self.speech_buffer.append(audio_chunk)
            self.silence_frames = 0
        else:
            # Update pre-roll buffer
            self.preroll_buffer.extend(audio_chunk)
            
            if self.is_speaking:
                self.silence_frames += 1
                self.speech_buffer.append(audio_chunk)  # Include trailing silence
                
                if self.silence_frames >= self.silence_threshold:
                    # Speech ended - return segment
                    segment = np.concatenate(self.speech_buffer)
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_frames = 0
                    return segment, prob
                    
        return None, prob
        
    def reset(self):
        """Reset buffer state."""
        self.preroll_buffer.clear()
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0


def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    """Generates audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n


def vad_filter_audio(audio_bytes: bytes, sample_rate=16000, confidence_threshold=0.5) -> bytes:
    """
    Returns only the speech segments of the audio.
    Input: raw bytes (int16 PCM)
    Output: raw bytes (int16 PCM)
    """
    vad = VoiceActivityDetector(sample_rate, confidence_threshold)
    
    # Convert to numpy for Silero
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Process in 30ms chunks
    chunk_size = int(sample_rate * 0.03)
    speech_chunks = []
    
    for i in range(0, len(audio_np) - chunk_size, chunk_size):
        chunk = audio_np[i:i + chunk_size]
        if vad.is_speech(chunk):
            speech_chunks.append(chunk)
            
    if not speech_chunks:
        return audio_bytes  # Return original if no speech detected
        
    result = np.concatenate(speech_chunks)
    return (result * 32768).astype(np.int16).tobytes()
