import torch
import numpy as np
import os
import json
import librosa
from src.database.db_manager import DBManager
from src.model.speaker_encoder import SpeakerEncoder
from src.audio.preprocessing import extract_mel_spectrogram

class SpeakerIdentifier:
    def __init__(self, model_path="models/speaker_encoder.pt", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpeakerEncoder().to(self.device)
        self.db = DBManager()
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model not found at {model_path}")

    def compute_embedding(self, audio_path, duration=1.0):
        try:
            # We can average multiple chunks for better robustness
            y_full, sr = librosa.load(audio_path, sr=8000)
            
            # Apply VAD before embedding generation too?
            # Yes, critical for inference accuracy
            from src.audio.preprocessing import apply_vad
            y_full = apply_vad(y_full, sr=sr)
            
            if len(y_full) == 0:
                print("Audio is purely silence.")
                return None

            # Create sliding windows of 1 sec
            chunk_len = int(duration * 8000)
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
             # Convert tensor to list/numpy for DB
            emb_list = embedding.cpu().tolist()
            if self.db.add_speaker(name, emb_list):
                print(f"Enrolled {name}.")
                return True
            else:
                print(f"Failed to enroll {name}.")
                return False
        return False

    def identify(self, audio_path, threshold=0.85):
        embedding = self.compute_embedding(audio_path)
        if embedding is None:
            return None
        
        # Determine strictness dynamically?
        # For now use fixed threshold.
        # DB returns closest match.
        
        emb_list = embedding.cpu().tolist()
        name, similarity = self.db.find_closest_speaker(emb_list)
        
        if name and similarity >= threshold:
            return {"speaker": name, "confidence": similarity}
        else:
            return {"speaker": "Unknown", "confidence": similarity}
