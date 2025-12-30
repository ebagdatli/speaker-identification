import torch
import numpy as np
import os
import json
import librosa
from src.model.speaker_encoder import SpeakerEncoder
from src.audio.preprocessing import extract_mel_spectrogram
from src.utils import config_loader as config

class SpeakerIdentifier:
    def __init__(self, model_path=None, embeddings_path=None, device=None):
        # Use config values as defaults
        self.model_path = model_path or config.model_path()
        self.embeddings_path = embeddings_path or config.embeddings_path()
        self.sample_rate = config.sample_rate()
        self.duration = config.duration()
        self.threshold = config.threshold()
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpeakerEncoder().to(self.device)
        self.speakers = {}
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model not found at {self.model_path}")
            
        self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'r') as f:
                data = json.load(f)
                self.speakers = {k: torch.tensor(v).to(self.device) for k, v in data.items()}

    def save_embeddings(self):
        data = {k: v.cpu().tolist() for k, v in self.speakers.items()}
        # Create dir if not exists
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, 'w') as f:
            json.dump(data, f)

    def compute_embedding(self, audio_path, duration=None):
        duration = duration or self.duration
        try:
            # We can average multiple chunks for better robustness
            y_full, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Create sliding windows
            chunk_len = int(duration * self.sample_rate)
            embeddings = []
            
            # If shorter than 1 sec, pad
            if len(y_full) < chunk_len:
                y = np.pad(y_full, (0, chunk_len - len(y_full)))
                y = y[:chunk_len]
                spec = extract_mel_spectrogram(y, sr=sr)
                t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embeddings.append(self.model(t))
            else:
                # Take up to 5 random chunks or sliding window
                # Sliding window with overlap
                stride = chunk_len // 2
                for start in range(0, len(y_full) - chunk_len, stride):
                    chunk = y_full[start:start+chunk_len]
                    spec = extract_mel_spectrogram(chunk, sr=sr)
                    t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embeddings.append(self.model(t))
                
                # If no chunks (perfectly equal??), take one
                if not embeddings:
                     y = y_full[:chunk_len]
                     spec = extract_mel_spectrogram(y, sr=sr)
                     t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device)
                     with torch.no_grad():
                        embeddings.append(self.model(t))

            # Stack and Average
            if len(embeddings) > 0:
                emb_tensor = torch.cat(embeddings, dim=0)
                # Average pooling
                avg_emb = torch.mean(emb_tensor, dim=0)
                # Re-normalize
                return torch.nn.functional.normalize(avg_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
            return None

        except Exception as e:
            print(f"Error computing embedding for {audio_path}: {e}")
            return None

    def enroll_speaker(self, name, audio_path):
        embedding = self.compute_embedding(audio_path)
        if embedding is not None:
            self.speakers[name] = embedding
            self.save_embeddings()
            print(f"Enrolled {name}.")
            return True
        return False

    def remove_speaker(self, name):
        if name in self.speakers:
            del self.speakers[name]
            self.save_embeddings()
            return True
        return False
        
    def refresh(self):
        """Reloads embeddings from disk."""
        self.load_embeddings()

    def identify(self, audio_path, threshold=None):
        threshold = threshold or self.threshold
        embedding = self.compute_embedding(audio_path)
        if embedding is None:
            return None
        
        best_score = -1.0
        best_speaker = None
        
        # Calculate similarities
        for name, ref_emb in self.speakers.items():
            score = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            # Debug: print(f"Score against {name}: {score}")
            if score > best_score:
                best_score = score
                best_speaker = name
        
        if best_score >= threshold:
            return {"speaker": best_speaker, "confidence": best_score}
        else:
            return {"speaker": "Unknown", "confidence": best_score}
