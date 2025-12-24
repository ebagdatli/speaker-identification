import os
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

def extract_mel_spectrogram(
    signal: np.ndarray, sr: int = 8000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 256
) -> np.ndarray:
    """
    Extracts log-power mel-spectrogram.
    Reused from deep-audio-fingerprinting project.
    """
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def load_audio_chunk(path: str, duration: float = 1.0, sr: int = 8000):
    """
    Load a random chunk of audio of specified duration.
    If audio is shorter, pad it.
    """
    try:
        # Get duration first to pick random offset
        total_duration = librosa.get_duration(filename=path)
        
        if total_duration < duration:
            offset = 0
            y, _ = librosa.load(path, sr=sr)
            # Pad with zeros
            target_len = int(duration * sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            y = y[:target_len]
        else:
            offset = random.uniform(0, total_duration - duration)
            y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
            
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(int(duration * sr))


class SpeakerDataset(Dataset):
    """
    Dataset for Speaker Identification using Triplet Loss.
    """
    def __init__(self, root_dir: str, sr: int = 8000, duration: float = 1.0):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.speakers = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.speaker_files = {}
        self.all_files = []
        
        for speaker in self.speakers:
            s_dir = os.path.join(root_dir, speaker)
            files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith('.wav')]
            if files:
                self.speaker_files[speaker] = files
                self.all_files.extend(files)
                
        # Filter speakers with no files
        self.speakers = [s for s in self.speakers if s in self.speaker_files]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # Anchor
        anchor_path = self.all_files[idx]
        # Identify speaker
        speaker = os.path.basename(os.path.dirname(anchor_path))
        
        # Positive: Same speaker
        # Try to pick a different file, if not possible, same file different chunk
        pos_files = self.speaker_files[speaker]
        if len(pos_files) > 1:
            pos_path = random.choice(pos_files) # Can be same file, that's ok (different chunk likely)
        else:
            pos_path = anchor_path
            
        # Negative: Different speaker
        neg_speaker = random.choice([s for s in self.speakers if s != speaker])
        neg_path = random.choice(self.speaker_files[neg_speaker])
        
        # Load audio
        y_anchor = load_audio_chunk(anchor_path, self.duration, self.sr)
        y_pos = load_audio_chunk(pos_path, self.duration, self.sr)
        y_neg = load_audio_chunk(neg_path, self.duration, self.sr)
        
        # Extract features
        s_anchor = extract_mel_spectrogram(y_anchor, sr=self.sr)
        s_pos = extract_mel_spectrogram(y_pos, sr=self.sr)
        s_neg = extract_mel_spectrogram(y_neg, sr=self.sr)
        
        return (
            torch.from_numpy(s_anchor).unsqueeze(0), # Add channel dim
            torch.from_numpy(s_pos).unsqueeze(0),
            torch.from_numpy(s_neg).unsqueeze(0),
            speaker # Label (optional, for debugging)
        )
