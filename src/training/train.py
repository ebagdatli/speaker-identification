import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio.preprocessing import SpeakerDataset
from src.model.speaker_encoder import SpeakerEncoder

def train(data_path, model_path, epochs=10, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist. Please create it and add data.")
        return

    dataset = SpeakerDataset(data_path)
    if len(dataset) == 0:
        print("No audio files found in dataset.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = SpeakerEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # batch: (anchor, pos, neg, labels)
            anchor, pos, neg, _ = batch
            
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            optimizer.zero_grad()
            
            embed_anchor = model(anchor)
            embed_pos = model(pos)
            embed_neg = model(neg)
            
            loss = criterion(embed_anchor, embed_pos, embed_neg)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        
        # Save checkpoint
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw data directory")
    parser.add_argument("--model_path", type=str, default="models/speaker_encoder.pt", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    train(args.data_path, args.model_path, args.epochs)
