import os
import psycopg2
# from pgvector.psycopg2 import register_vector # Removed for fallback
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DBManager:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.dbname = os.getenv("DB_NAME", "speaker_db")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "yourpassword")
        self.conn = None

    def connect(self):
        """Calculates connection string and establishes a connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            # register_vector(self.conn) # Removed
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise e

    def init_db(self):
        """Initializes the database schema."""
        if not self.conn:
            self.connect()
        
        try:
            cur = self.conn.cursor()
            # We skip CREATE EXTENSION assuming user doesn't have it installed on OS level
            # cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create speakers table
            # Changed vector(256) to float8[] for compatibility
            cur.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    embedding float8[], 
                    sample_count INT DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
            cur.close()
            print("Database initialized (Using float8[] storage).")
        except Exception as e:
            print(f"Error initializing database: {e}")
            self.conn.rollback()

    def add_speaker(self, name, embedding):
        """Adds a new speaker or updates existing one by averaging embeddings."""
        if not self.conn:
            self.connect()

        name = name.strip() # Sanitize
        
        try:
            cur = self.conn.cursor()
            
            # Helper to perform the update logic
            def do_update(curr_emb_list, curr_count):
                new_vec = np.array(embedding, dtype=np.float32)
                curr_vec = np.array(curr_emb_list, dtype=np.float32)
                
                # Weighted Avg
                updated_vec = (curr_vec * curr_count + new_vec) / (curr_count + 1)
                norm = np.linalg.norm(updated_vec)
                if norm > 0:
                    updated_vec = updated_vec / norm
                    
                new_cnt = curr_count + 1
                
                cur.execute("""
                    UPDATE speakers 
                    SET embedding = %s, sample_count = %s, created_at = CURRENT_TIMESTAMP 
                    WHERE name = %s
                """, (updated_vec.tolist(), new_cnt, name))
                return new_cnt

            # 1. Try to find existing
            cur.execute("SELECT embedding, sample_count FROM speakers WHERE name = %s", (name,))
            row = cur.fetchone()
            
            if row:
                # Update existing
                current_emb = row[0]
                current_count = row[1] if row[1] is not None else 1
                new_c = do_update(current_emb, current_count)
                print(f"Updated {name} (Samples: {new_c})")
            else:
                # 2. Try Insert
                try:
                    new_vec_list = np.array(embedding, dtype=np.float32).tolist()
                    cur.execute("""
                        INSERT INTO speakers (name, embedding, sample_count) 
                        VALUES (%s, %s, 1)
                    """, (name, new_vec_list))
                    print(f"Inserted new speaker {name}")
                except psycopg2.errors.UniqueViolation:
                    # Race condition: It was inserted by someone else (or previous loop step?) just now
                    self.conn.rollback() # Must rollback current transaction block
                    
                    # Start new transaction logic
                    # We need to Select again
                    cur = self.conn.cursor() 
                    cur.execute("SELECT embedding, sample_count FROM speakers WHERE name = %s", (name,))
                    row2 = cur.fetchone()
                    if row2:
                        current_emb = row2[0]
                        current_count = row2[1] if row2[1] is not None else 1
                        new_c = do_update(current_emb, current_count)
                        print(f"Recovered & Updated {name} (Samples: {new_c})")
                    else:
                        print(f"Error: UniqueViolation but could not find row for {name}")
                        return False

            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            print(f"Error adding speaker {name}: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def find_closest_speaker(self, embedding, limit=1):
        """Finds the closest speaker using cosine distance (Calculated in Python)."""
        if not self.conn:
            self.connect()
            
        try:
            cur = self.conn.cursor()
            
            # Fetch ALL speakers (Not efficient for millions, but fine for thousands)
            cur.execute("SELECT name, embedding FROM speakers;")
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None, 0.0

            # Perform Cosine Similarity in Python
            # embedding is the query vector (list or numpy array)
            query_vec = np.array(embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return None, 0.0
            
            best_score = -1.0
            best_name = None
            
            for name, db_emb_list in rows:
                if db_emb_list is None:
                    continue
                db_vec = np.array(db_emb_list, dtype=np.float32)
                db_norm = np.linalg.norm(db_vec)
                
                if db_norm == 0:
                    continue
                    
                # Cosine Similarity = (A . B) / (||A|| * ||B||)
                score = np.dot(query_vec, db_vec) / (query_norm * db_norm)
                
                if score > best_score:
                    best_score = score
                    best_name = name
            
            return best_name, float(best_score)
            
        except Exception as e:
            print(f"Error finding speaker: {e}")
            self.conn.rollback()
            return None, 0.0

    def close(self):
        if self.conn:
            self.conn.close()
