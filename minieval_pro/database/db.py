import sqlite3
from datetime import datetime
import os
import uuid

# Get the project root directory
# Database location: current working directory by default,
# overridable with the MINIEVAL_DB_PATH environment variable.
DB_PATH = os.environ.get("MINIEVAL_DB_PATH", os.path.join(os.getcwd(), "minieval.db"))

def init_db():
    """Initialize the database with the evaluations table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            context TEXT,
            faithfulness REAL,
            relevance REAL,
            toxicity REAL,
            overall_score REAL,
            passed BOOLEAN,
            failure_reason TEXT,
            model_name TEXT,
            model_temperature REAL,
            prompt_template TEXT,
            evaluation_duration REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[MiniEval] Database initialized at {DB_PATH}")
    ...

def save_evaluation(
    question, answer, context, 
    faithfulness, relevance, toxicity, 
    overall_score, passed, failure_reason=None,
    model_name=None, model_temperature=None, 
    prompt_template=None, evaluation_duration=None
):
    """Save a single evaluation to the database"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Generate unique run_id
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    run_id = f"ev_{timestamp_str}_{unique_id}"
    
    cursor.execute('''
        INSERT INTO evaluations (
            run_id, timestamp, question, answer, context, 
            faithfulness, relevance, toxicity, overall_score, 
            passed, failure_reason, model_name, model_temperature, 
            prompt_template, evaluation_duration
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id,
        datetime.now().isoformat(),
        question,
        answer,
        context,
        faithfulness,
        relevance,
        toxicity,
        overall_score,
        passed,
        failure_reason,
        model_name,
        model_temperature,
        prompt_template,
        evaluation_duration
    ))
    
    eval_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return eval_id, run_id

def get_recent_evaluations(limit=100):
    """Get the most recent evaluations"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, question, answer, overall_score, passed, failure_reason
        FROM evaluations
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    evaluations = []
    for row in rows:
        evaluations.append({
            'id': row[0],
            'timestamp': row[1],
            'question': row[2],
            'answer': row[3],
            'overall_score': row[4],
            'passed': bool(row[5]),
            'failure_reason': row[6]
        })
    
    return evaluations

def get_evaluation_by_id(eval_id):
    """Get a single evaluation by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            id, run_id, timestamp, question, answer, context, 
            faithfulness, relevance, toxicity, overall_score, 
            passed, failure_reason, model_name, model_temperature, 
            prompt_template, evaluation_duration
        FROM evaluations 
        WHERE id = ?
    ''', (eval_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'run_id': row[1],
            'timestamp': row[2],
            'question': row[3],
            'answer': row[4],
            'context': row[5],
            'faithfulness': row[6],
            'relevance': row[7],
            'toxicity': row[8],
            'overall_score': row[9],
            'passed': row[10],
            'failure_reason': row[11],
            'model_name': row[12],
            'model_temperature': row[13],
            'prompt_template': row[14],
            'evaluation_duration': row[15]
        }
    return None
def get_correlation_metrics():
    """Get correlation data between MiniEval and GPT-4"""
    import json
    import os
    
    metrics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'correlation_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None