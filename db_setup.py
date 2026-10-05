# db_setup.py
import sqlite3
import os

def init_db():
    db_name = 'defects_log.db'
    
    # Connect to SQLite (this automatically creates the file if it doesn't exist)
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Create the logging table matching the shared fusion record schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            rgb_defects TEXT,
            ae_state TEXT,
            confidence REAL NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Local Edge Ledger Matrix Initialized: '{db_name}'")
    print("🚀 Table 'inspection_logs' is configured and ready to log fusion data.")

if __name__ == "__main__":
    init_db()