import torch
import numpy as np
import os
import json
import librosa
from src.model.speaker_encoder import SpeakerEncoder
from src.audio.preprocessing import extract_mel_spectrogram

class SpeakerIdentifier:
    def __init__(self, model_path="models/speaker_encoder.pt", embeddings_path="embeddings/speakers.json", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpeakerEncoder().to(self.device)
        self.embeddings_path = embeddings_path
        self.speakers = {}
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print(f"Warning: Model not found at {model_path}. Using random weights.")
            
        self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'r') as f:
                data = json.load(f)
                # Convert list back to tensor
                self.speakers = {k: torch.tensor(v).to(self.device) for k, v in data.items()}

    def save_embeddings(self):
        # Convert tensors to list for JSON serialization
        data = {k: v.cpu().tolist() for k, v in self.speakers.items()}
        with open(self.embeddings_path, 'w') as f:
            json.dump(data, f)

    def compute_embedding(self, audio_path):
        # Load audio (can refer to preprocessing.load_audio_chunk but better to take full audio or average chunks)
        # For simplicity, let's take the first 1 second or pad
        try:
            y, sr = librosa.load(audio_path, sr=8000, duration=1.0) # Using 1s for consistency with training
            target_len = int(1.0 * 8000)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            y = y[:target_len]
            
            spec = extract_mel_spectrogram(y, sr=8000)
            tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 256, 32)
            
            with torch.no_grad():
                embedding = self.model(tensor)
            return embedding.squeeze(0) # (256,)
        except Exception as e:
            print(f"Error computing embedding for {audio_path}: {e}")
            return None

    def enroll_speaker(self, name, audio_path):
        embedding = self.compute_embedding(audio_path)
        if embedding is not None:
            # If speaker exists, average? Or overwrite? Let's overwrite or add to list. 
            # Simple version: overwrite
            self.speakers[name] = embedding
            self.save_embeddings()
            print(f"Enrolled {name}.")
            return True
        return False

    def identify(self, audio_path, threshold=0.8):
        embedding = self.compute_embedding(audio_path)
        if embedding is None:
            return None
        
        best_score = -1.0
        best_speaker = None
        
        for name, ref_emb in self.speakers.items():
            # Cosine similarity
            score = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            if score > best_score:
                best_score = score
                best_speaker = name
        
        if best_score >= threshold:
            return {"speaker": best_speaker, "confidence": best_score}
        else:
            return {"speaker": None, "confidence": best_score}
