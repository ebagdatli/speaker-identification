import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio.preprocessing import SpeakerDataset
from src.model.speaker_encoder import SpeakerEncoder
from src.training.loss import NTxent_Loss

def train(data_path, model_path, epochs=20, batch_size=32, lr=0.0005, noise_path=None, ir_path=None):
    # Convert to absolute paths to avoid issues with relative paths in torch.save
    data_path = os.path.abspath(data_path)
    model_path = os.path.abspath(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return

    # Pass augmentation paths if they exist
    dataset = SpeakerDataset(data_path, noise_path=noise_path, ir_path=ir_path)
    if len(dataset) == 0:
        print("No audio files found.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Model
    model = SpeakerEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss: Contrastive
    criterion = NTxent_Loss(temperature=0.1)
    
    print(f"Starting training with NTxent Loss for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # batch: (view1, view2, label_debug)
            view1, view2, _ = batch
            
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            optimizer.zero_grad()
            
            embed1 = model(view1)
            embed2 = model(view2)
            
            loss = criterion(embed1, embed2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Save checkpoint
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/speaker_encoder_1.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--noise_path", type=str, default=None, help="Path to background noise used for augmentation")
    parser.add_argument("--ir_path", type=str, default=None, help="Path to impulse responses")
    
    args = parser.parse_args()
    
    train(args.data_path, args.model_path, args.epochs, noise_path=args.noise_path, ir_path=args.ir_path)
