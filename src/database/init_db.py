import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.database.db_manager import DBManager

def main():
    print("Initializing database...")
    try:
        db = DBManager()
        db.init_db()
        print("Done.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")

if __name__ == "__main__":
    main()
