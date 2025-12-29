import sys
import os
import torch
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.inference.identify_speaker import SpeakerIdentifier
from src.database.db_manager import DBManager

def debug_match(audio_path):
    print(f"\n--- Debugging: {audio_path} ---")
    
    # 1. Initialize Identifier
    try:
        identifier = SpeakerIdentifier()
    except Exception as e:
        print(f"Error initializing identifier: {e}")
        return

    # 2. Check Determinism
    print("\n[Adım 1] Determinizm Kontrolü (Aynı dosya 2 kez işleniyor)")
    emb1 = identifier.compute_embedding(audio_path)
    emb2 = identifier.compute_embedding(audio_path)
    
    if emb1 is None or emb2 is None:
        print("HATA: Embedding üretilemedi (Sessiz dosya?).")
        return

    # Cosine sim between two runs
    sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    print(f"Embedding 1 vs Embedding 2 Benzerliği: {sim:.6f}")
    if sim < 0.9999:
        print("UYARI: Embedding üretimi deterministik değil! Model her seferinde farklı çıktı veriyor.")
    else:
        print("BAŞARILI: Embedding üretimi kararlı.")

    # 3. Check DB Scores
    print("\n[Adım 2] Veritabanı Skorları")
    db = DBManager()
    db.connect()
    cur = db.conn.cursor()
    cur.execute("SELECT name, embedding, sample_count FROM speakers;")
    rows = cur.fetchall()
    cur.close()
    db.close()

    query_vec = emb1.cpu().numpy()
    query_norm = np.linalg.norm(query_vec)

    scores = []
    for name, db_emb_list, count in rows:
        if db_emb_list is None:
            continue
        db_vec = np.array(db_emb_list, dtype=np.float32)
        db_norm = np.linalg.norm(db_vec)
        
        score = np.dot(query_vec, db_vec) / (query_norm * db_norm)
        scores.append((name, score, count))

    # Sort by score desc
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"{'İsim':<30} {'Benzerlik':<10} {'Örnek Sayısı'}")
    print("-" * 60)
    for name, score, count in scores:
        print(f"{name:<30} {score:.4f}     {count}")

    print("\n--------------------------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python src/debug/debug_match.py <audio_file_path>")
    else:
        debug_match(sys.argv[1])
