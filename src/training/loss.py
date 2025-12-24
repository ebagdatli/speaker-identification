import torch
import torch.nn as nn
import torch.nn.functional as F

class NTxent_Loss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Ported from deep-audio-fingerprinting.
    """
    def __init__(self, temperature: float = 0.5):
        super(NTxent_Loss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, out_1, out_2):
        """
        out_1: (B, D) - First view
        out_2: (B, D) - Second view (augmented)
        """
        batch_size = out_1.shape[0]
        
        # Concatenate: [View1_1, ..., View1_B, View2_1, ..., View2_B]
        out = torch.cat([out_1, out_2], dim=0) # (2B, D)
        
        # Similarity matrix
        # sim_matrix[i, j] = cosine similarity between out[i] and out[j]
        # Assuming vectors are already normalized
        sim_matrix = torch.matmul(out, out.t()) / self.temperature
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(out.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs:
        # (i, i + B) and (i + B, i)
        # Create labels
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(out.device)
        
        loss = self.cross_entropy(sim_matrix, labels)
        return loss
