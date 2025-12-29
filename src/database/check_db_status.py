import sys
import os
import psycopg2
from dotenv import load_dotenv

# Load env from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.database.db_manager import DBManager

def check_speakers():
    db = DBManager()
    db.connect()
    
    try:
        cur = db.conn.cursor()
        cur.execute("SELECT id, name, sample_count, created_at FROM speakers;")
        rows = cur.fetchall()
        
        print("\n--- Veritabanı Durumu (Speakers Tablosu) ---")
        print(f"{'ID':<5} {'İsim':<30} {'Örnek Sayısı (Sample Count)':<15} {'Oluşturulma Tarihi'}")
        print("-" * 80)
        
        for row in rows:
            # Handle potential None for sample_count if migration failed silently or old rows exist
            s_count = row[2] if row[2] is not None else "NULL"
            print(f"{row[0]:<5} {row[1]:<30} {s_count:<15} {row[3]}")
            
        print("-" * 80)
        
    except Exception as e:
        print(f"Error checking DB: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_speakers()
