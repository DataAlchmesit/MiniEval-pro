"""Run the MiniEval dashboard"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from minieval.database.db import init_db

if __name__ == "__main__":
    # Initialize the database
    print("Initializing MiniEval database...")
    init_db()
    
    print("Starting MiniEval dashboard at http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    # Run the FastAPI server - FIXED PATH
    uvicorn.run("minieval.web.app:app", host="127.0.0.1", port=8000, reload=True)