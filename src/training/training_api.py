"""
Training API Module

UI'dan çağrılabilir training işlemleri için wrapper modülü.
"""

import os
import shutil
import tempfile
import threading
from typing import Dict, List, Callable, Optional
import torch

# Training status - thread-safe
_training_status = {
    "is_running": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "status_message": "",
    "completed": False,
    "error": None
}
_status_lock = threading.Lock()


def get_training_status() -> dict:
    """Returns current training status."""
    with _status_lock:
        return _training_status.copy()


def _update_status(**kwargs):
    """Thread-safe status update."""
    with _status_lock:
        _training_status.update(kwargs)


def prepare_training_data(speakers_data: Dict[str, List[bytes]], base_path: str = "data/raw") -> str:
    """
    Saves uploaded audio files to the training data directory.
    
    Args:
        speakers_data: Dict mapping speaker names to list of audio file bytes
        base_path: Base directory for training data
        
    Returns:
        Path to the prepared data directory
    """
    base_path = os.path.abspath(base_path)
    os.makedirs(base_path, exist_ok=True)
    
    saved_count = 0
    
    for speaker_name, audio_files in speakers_data.items():
        # Sanitize speaker name for directory
        safe_name = speaker_name.strip().upper()
        speaker_dir = os.path.join(base_path, safe_name)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Get existing file count to avoid overwriting
        existing_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
        start_idx = len(existing_files) + 1
        
        for i, audio_bytes in enumerate(audio_files):
            file_name = f"{start_idx + i:03d}.wav"
            file_path = os.path.join(speaker_dir, file_name)
            
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)
            saved_count += 1
    
    return base_path


def start_training_async(
    data_path: str,
    model_path: str = "models/speaker_encoder.pt",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    pretrained_model: Optional[str] = None
) -> bool:
    """
    Starts training in a background thread.
    
    Args:
        data_path: Path to training data
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        pretrained_model: Path to pretrained model for fine-tuning
        
    Returns:
        True if training started successfully
    """
    status = get_training_status()
    if status["is_running"]:
        return False
    
    # Reset status
    _update_status(
        is_running=True,
        current_epoch=0,
        total_epochs=epochs,
        current_loss=0.0,
        status_message="Eğitim başlatılıyor...",
        completed=False,
        error=None
    )
    
    # Start training in background thread
    thread = threading.Thread(
        target=_training_worker,
        args=(data_path, model_path, epochs, batch_size, learning_rate, pretrained_model),
        daemon=True
    )
    thread.start()
    
    return True


def _training_worker(
    data_path: str,
    model_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained_model: Optional[str]
):
    """Background training worker."""
    try:
        from src.audio.preprocessing import SpeakerDataset
        from src.model.speaker_encoder import SpeakerEncoder
        from src.training.loss import NTxent_Loss
        from torch.utils.data import DataLoader
        import torch.optim as optim
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _update_status(status_message=f"Cihaz: {device}")
        
        # Load dataset
        _update_status(status_message="Veri seti yükleniyor...")
        dataset = SpeakerDataset(data_path)
        
        if len(dataset) == 0:
            _update_status(
                is_running=False,
                error="Veri bulunamadı. Lütfen ses dosyalarını yükleyin.",
                status_message="Hata: Veri bulunamadı"
            )
            return
        
        if len(dataset) < batch_size:
            batch_size = max(2, len(dataset) // 2)
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        if len(dataloader) == 0:
            _update_status(
                is_running=False,
                error="Yeterli veri yok. En az 2 konuşmacı ve toplam 4 ses dosyası gerekli.",
                status_message="Hata: Yetersiz veri"
            )
            return
        
        # Initialize model
        _update_status(status_message="Model oluşturuluyor...")
        model = SpeakerEncoder().to(device)
        
        # Load pretrained weights for fine-tuning
        if pretrained_model and os.path.exists(pretrained_model):
            try:
                model.load_state_dict(torch.load(pretrained_model, map_location=device))
                _update_status(status_message="Önceki model yüklendi (fine-tuning)")
            except Exception as e:
                _update_status(status_message=f"Uyarı: Model yüklenemedi, sıfırdan başlanıyor")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        criterion = NTxent_Loss(temperature=0.1)
        
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in dataloader:
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
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            scheduler.step(avg_loss)
            
            _update_status(
                current_epoch=epoch + 1,
                current_loss=avg_loss,
                status_message=f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
            )
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
        
        _update_status(
            is_running=False,
            completed=True,
            status_message=f"Eğitim tamamlandı! En iyi loss: {best_loss:.4f}"
        )
        
    except Exception as e:
        _update_status(
            is_running=False,
            error=str(e),
            status_message=f"Hata: {str(e)}"
        )


def enroll_all_speakers(data_path: str, model_path: str = "models/speaker_encoder.pt") -> Dict[str, bool]:
    """
    Enrolls all speakers from the training data directory to the database.
    
    Args:
        data_path: Path to training data with speaker subdirectories
        model_path: Path to the trained model
        
    Returns:
        Dict mapping speaker names to enrollment success status
    """
    from src.inference.identify_speaker import SpeakerIdentifier
    
    try:
        identifier = SpeakerIdentifier(model_path=model_path)
    except Exception as e:
        raise Exception(f"Model yüklenemedi: {e}")
    
    results = {}
    
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        raise Exception(f"Veri klasörü bulunamadı: {data_path}")
    
    for speaker_name in os.listdir(data_path):
        speaker_dir = os.path.join(data_path, speaker_name)
        if not os.path.isdir(speaker_dir):
            continue
            
        wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
        if not wav_files:
            results[speaker_name] = False
            continue
        
        success_count = 0
        for wav_file in wav_files:
            file_path = os.path.join(speaker_dir, wav_file)
            try:
                if identifier.enroll_speaker(speaker_name, file_path):
                    success_count += 1
            except Exception as e:
                print(f"Error enrolling {speaker_name} with {wav_file}: {e}")
        
        results[speaker_name] = success_count > 0
    
    return results
