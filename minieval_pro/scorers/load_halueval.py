import sqlite3
from datasets import load_dataset

print("=" * 60)
print("🔄 Loading HaluEval Dataset into MiniEval Database")
print("=" * 60)

# Load HaluEval dataset
print("\n📥 Downloading HaluEval dataset...")
dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
print(f"✅ Loaded {len(dataset)} samples")

# Connect to database
print("\n💾 Connecting to minieval.db...")
conn = sqlite3.connect('minieval.db')
cursor = conn.cursor()

# Create halueval_samples table
print("📋 Creating halueval_samples table...")
cursor.execute('''
    CREATE TABLE IF NOT EXISTS halueval_samples (
        id INTEGER PRIMARY KEY,
        question TEXT,
        answer TEXT,
        ground_truth TEXT,
        knowledge_context TEXT,
        minieval_prediction TEXT,
        confidence REAL,
        is_correct INTEGER,
        evaluated INTEGER DEFAULT 0,
        evaluation_timestamp DATETIME
    )
''')

# Check if data already exists
cursor.execute("SELECT COUNT(*) FROM halueval_samples")
count = cursor.fetchone()[0]

if count > 0:
    print(f"⚠️ Database already has {count} samples")
    response = input("Do you want to reload all data? (y/n): ")
    if response.lower() == 'y':
        cursor.execute("DELETE FROM halueval_samples")
        conn.commit()
        print("✅ Cleared existing data")
        count = 0

# Load data if table is empty
if count == 0:
    print("\n🔄 Loading HaluEval samples into database...")
    
    for idx, sample in enumerate(dataset):
        cursor.execute('''
            INSERT INTO halueval_samples 
            (id, question, answer, ground_truth, knowledge_context, evaluated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            idx,
            sample['question'],
            sample['answer'],
            sample['hallucination'],
            sample['knowledge'][:2000],  # Limit to 2000 chars
            0  # Not evaluated yet
        ))
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            conn.commit()
            print(f"  ✅ Loaded {idx + 1}/{len(dataset)} samples")
    
    conn.commit()
    print(f"\n✅ Successfully loaded {len(dataset)} samples into database!")
    
    # Show statistics
    cursor.execute("SELECT ground_truth, COUNT(*) FROM halueval_samples GROUP BY ground_truth")
    stats = cursor.fetchall()
    print("\n📊 Dataset Statistics:")
    for label, count in stats:
        label_name = "Hallucinated" if label == 'yes' else "Non-Hallucinated"
        print(f"  • {label_name}: {count} samples")
else:
    print(f"✅ Database already has {count} samples ready!")

# Verify data was loaded
cursor.execute("SELECT COUNT(*) FROM halueval_samples")
final_count = cursor.fetchone()[0]
print(f"\n📊 Total samples in database: {final_count}")

# Show a sample
cursor.execute("SELECT id, question, ground_truth FROM halueval_samples LIMIT 1")
sample = cursor.fetchone()
if sample:
    print(f"\n📝 Sample record:")
    print(f"  ID: {sample[0]}")
    print(f"  Question: {sample[1][:100]}...")
    print(f"  Ground Truth: {'Hallucinated' if sample[2] == 'yes' else 'Not Hallucinated'}")

conn.close()
print("\n" + "=" * 60)
print("✅ Database setup complete!")
print("=" * 60)
print("\n🎯 Next Steps:")
print("1. Restart your dashboard: python run_dashboard.py")
print("2. Open: http://localhost:8000/live_feed")
print("3. The dashboard should now show HaluEval samples")