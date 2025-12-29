import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.database.db_manager import DBManager

def migrate():
    print("Migrating database...")
    db = DBManager()
    db.connect()
    try:
        cur = db.conn.cursor()
        cur.execute("ALTER TABLE speakers ADD COLUMN IF NOT EXISTS sample_count INT DEFAULT 1;")
        db.conn.commit()
        cur.close()
        print("Migration successful: Added sample_count column.")
    except Exception as e:
        print(f"Migration failed: {e}")
        db.conn.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
