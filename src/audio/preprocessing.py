import os
import random
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
import warnings

# Config loader
try:
    from src.utils import config_loader as config
except ImportError:
    config = None

try:
    from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, HighPassFilter, LowPassFilter, TimeStretch, PitchShift
except ImportError:
    warnings.warn("audiomentations not found. Data augmentation will be disabled.", UserWarning)
    Compose = None


# --- Augmentation Utils (Ported) ---

def add_background_noise(y: np.ndarray, y_noise: np.ndarray, SNR: float) -> np.ndarray:
    if y.size < y_noise.size:
        y_noise = y_noise[:y.size]
    else:
        y_noise = np.resize(y_noise, y.shape)
    
    snr = 10**(SNR / 10)
    E_y = np.sum(y**2)
    # Avoid division by zero
    if E_y == 0:
        return y
        
    E_n = np.sum(y_noise**2)
    if E_n == 0:
        return y

    z = np.sqrt((E_n / E_y) * snr) * y + y_noise
    return z / (np.max(np.abs(z)) + 1e-9)

def time_offset_modulation(signal: np.ndarray, time_index: int, sr: int = 8000, max_offset: float = 0.25) -> np.ndarray:
    # Simplified version that just shifts
    offset_samples = int(random.uniform(-max_offset, max_offset) * sr)
    if offset_samples == 0:
        return signal
    
    # Shift and pad/crop
    if offset_samples > 0:
        return np.pad(signal, (offset_samples, 0), mode='constant')[:len(signal)]
    else:
        return np.pad(signal, (0, -offset_samples), mode='constant')[-len(signal):]

def extract_mel_spectrogram(
    signal: np.ndarray, sr: int = None, n_fft: int = None, hop_length: int = None, n_mels: int = None
) -> np.ndarray:
    # Use config defaults
    sr = sr or (config.get('audio.sample_rate', 8000) if config else 8000)
    n_fft = n_fft or (config.get('audio.n_fft', 1024) if config else 1024)
    hop_length = hop_length or (config.get('audio.hop_length', 256) if config else 256)
    n_mels = n_mels or (config.get('audio.n_mels', 256) if config else 256)
    
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def audio_augmentation_chain(
    signal: np.ndarray, 
    rng: np.random.Generator, 
    noise_path: str = None, 
    ir_path: str = None, 
    sr: int = 8000
):
    """
    Applies augmentations to a signal.
    """
    augmented = signal.copy()
    
    # Early return if no augmentation library
    if Compose is None:
        return augmented

    transforms = []
    
    # 1. Background Noise / Impulse Response
    if noise_path and os.path.exists(noise_path):
        transforms.append(AddBackgroundNoise(sounds_path=noise_path, min_snr_in_db=5, max_snr_in_db=20, p=0.6))
    
    if ir_path and os.path.exists(ir_path):
        transforms.append(ApplyImpulseResponse(ir_path=ir_path, p=0.4))
        
    # 2. General Audio degradations
    transforms.extend([
        # Pitch shifting (simulate different intonation)
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        # Time stretching (simulate speed variance)
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    ])
    
    if transforms:
        augmenter = Compose(transforms)
        try:
            # audiomentations expects float32
            if augmented.dtype != np.float32:
                augmented = augmented.astype(np.float32)
            augmented = augmenter(samples=augmented, sample_rate=sr)
        except Exception as e:
            # print(f"Augmentation failed: {e}")
            pass

    return augmented

# --- Dataset ---

def load_audio_chunk(path: str, duration: float = 1.0, sr: int = 8000):
    try:
        total_duration = librosa.get_duration(filename=path)
        if total_duration < duration:
            y, _ = librosa.load(path, sr=sr)
            target_len = int(duration * sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            y = y[:target_len]
        else:
            offset = random.uniform(0, total_duration - duration)
            y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
        return y
    except Exception as e:
        # print(f"Error loading {path}: {e}")
        return np.zeros(int(duration * sr))

class SpeakerDataset(Dataset):
    """
    Dataset for Contrastive Learning.
    Yields (View1, View2) where both are from the same speaker.
    View2 is heavily augmented or a different chunk.
    """
    def __init__(self, root_dir: str = None, file_list: list = None, noise_path: str = None, ir_path: str = None, sr: int = 8000, duration: float = 1.0):
        self.root_dir = root_dir
        self.noise_path = noise_path
        self.ir_path = ir_path
        self.sr = sr
        self.duration = duration
        self.rng = np.random.default_rng()
        
        self.speaker_files = {}
        self.all_files = []
        
        # Mode 1: Pass list of files directly (e.g. for validation split)
        if file_list is not None:
             self.all_files = file_list
             for f in file_list:
                 # Assumption: folder structure is name/file.wav
                 speaker = os.path.basename(os.path.dirname(f))
                 if speaker not in self.speaker_files:
                     self.speaker_files[speaker] = []
                 self.speaker_files[speaker].append(f)
             self.speakers = list(self.speaker_files.keys())

        # Mode 2: Scan directory
        elif root_dir is not None and os.path.exists(root_dir):
            self.speakers = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            for speaker in self.speakers:
                s_dir = os.path.join(root_dir, speaker)
                files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith('.wav')]
                if files:
                    self.speaker_files[speaker] = files
                    self.all_files.extend(files)
        else:
            print("Error: Either root_dir or file_list must be provided to SpeakerDataset")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # Anchor file
        file1 = self.all_files[idx]
        speaker = os.path.basename(os.path.dirname(file1))
        
        # Pick another file from same speaker for View 2 (or same file different chunk)
        # To make it robust to "same voice different session", prefer different file if possible
        candidates = self.speaker_files[speaker]
        if len(candidates) > 1:
            file2 = random.choice(candidates) # Might be same as file1, that's ok
        else:
            file2 = file1
            
        y1 = load_audio_chunk(file1, self.duration, self.sr)
        y2 = load_audio_chunk(file2, self.duration, self.sr)
        
        # Apply Augmentation to View 2 (and maybe View 1 lightly?)
        # Let's keep View 1 relatively clean or lightly augmented, View 2 heavily augmented
        
        # Augment y2
        y2_aug = audio_augmentation_chain(y2, self.rng, self.noise_path, self.ir_path, self.sr)
        
        # Sometimes augment y1 too, to avoid model learning "Clean vs Noisy" shortcut
        if self.rng.random() > 0.5:
             y1 = audio_augmentation_chain(y1, self.rng, self.noise_path, self.ir_path, self.sr)

        s1 = extract_mel_spectrogram(y1, sr=self.sr)
        s2 = extract_mel_spectrogram(y2_aug, sr=self.sr)
        
        return (
            torch.from_numpy(s1).unsqueeze(0),
            torch.from_numpy(s2).unsqueeze(0),
            speaker
        )
