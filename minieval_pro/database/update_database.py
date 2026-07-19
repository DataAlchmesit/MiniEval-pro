import sqlite3

conn = sqlite3.connect('minieval.db')
cursor = conn.cursor()

# Create datasets table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        filename TEXT,
        total_samples INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 0
    )
''')

# Create evaluations table with dataset_id
cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        question TEXT,
        answer TEXT,
        ground_truth TEXT,
        faithfulness_score REAL,
        relevance_score REAL,
        overall_score REAL,
        status TEXT,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
''')

# Check if old evaluations table exists and migrate data
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations'")
if cursor.fetchone():
    # Create default dataset for old data
    cursor.execute('''
        INSERT OR IGNORE INTO datasets (name, description, is_active)
        VALUES ('Legacy Evaluations', 'Imported from previous evaluations', 1)
    ''')
    cursor.execute("SELECT id FROM datasets WHERE name='Legacy Evaluations'")
    dataset_id = cursor.fetchone()[0]
    
    # Migrate old data
    cursor.execute('''
        INSERT INTO evaluation_results (dataset_id, question, answer, faithfulness_score, 
        relevance_score, overall_score, status, evaluated_at)
        SELECT ?, question, answer, faithfulness_score, relevance_score, 
        overall_score, status, evaluation_date
        FROM evaluations
    ''', (dataset_id,))
    print("✅ Migrated legacy data")

conn.commit()
conn.close()
print("✅ Database updated with dataset management features")