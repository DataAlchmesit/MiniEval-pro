"""
MiniEval + TruthfulQA Test Script
Tests your LLM against the TruthfulQA benchmark and evaluates with MiniEval
"""

import sys
import os
import csv
from datetime import datetime
import requests

# Add project path
sys.path.insert(0, r'C:\Users\PREETI SONI\minieval_project')

# Import MiniEval
from minieval import Evaluator

# Import dataset library
from datasets import load_dataset

# ============================================
# OLLAMA LLM FUNCTION
# ============================================

def get_llm_answer(question: str) -> str:
    """Get answer from Ollama local LLM"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama3.2:1b",
                "prompt": question,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response field')
        else:
            return f"HTTP Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running."
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================
# MAIN TEST FUNCTION
# ============================================

def main():
    print("=" * 60)
    print(" MiniEval + TruthfulQA Test Suite")
    print("=" * 60)
    
    # Test Ollama connection first
    print("\n🔌 Testing Ollama connection...")
    test_response = get_llm_answer("Say 'OK'")
    
    if test_response.startswith("Error"):
        print(f"   ❌ {test_response}")
        print("\n   To fix:")
        print("   1. Make sure Ollama is installed")
        print("   2. Run: ollama pull llama3.2:1b")
        print("   3. Keep Ollama running in background")
        return
    else:
        print(f"   Ollama connected! Response: {test_response[:50]}...")
    
    # Load TruthfulQA dataset
    print("\n Loading TruthfulQA dataset...")
    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        print(f" Loaded {len(dataset)} total questions")
    except Exception as e:
        print(f" Error loading dataset: {e}")
        print("   Try running: pip install datasets")
        return
    
    # Limit questions for quick testing (change to None for full test)
    test_limit = 5
    if test_limit:
        test_batch = dataset.select(range(min(test_limit, len(dataset))))
        print(f" Testing first {len(test_batch)} questions")
    else:
        test_batch = dataset
        print(f" Testing all {len(test_batch)} questions")
    
    print("\n🔧 Initializing MiniEval...")
    ev = Evaluator()
    
    results = []
    print("\n" + "=" * 60)
    
    for i, item in enumerate(test_batch):
        question = item['question']
        correct_answer = item['correct_answers'][0] if item['correct_answers'] else "No correct answer provided"
        
        print(f"\n Test {i+1}/{len(test_batch)}")
        print(f"   Question: {question[:80]}...")
        
        # Get LLM answer
        print(f"    Asking LLM...")
        llm_output = get_llm_answer(question)
        print(f"    LLM Answer: {llm_output[:100]}...")
        
        # Evaluate with MiniEval
        print(f"    Evaluating with MiniEval...")
        result = ev.score(
            question=question,
            context=correct_answer,
            answer=llm_output
        )
        
        status = " PASS" if result.passed() else "❌ FAIL"
        print(f"    Score: {result.overall:.2f} - {status}")
        
        # Get summary instead of failure_reason (since EvalResult doesn't have it)
        summary = result.summary()
        print(f"    Summary: {summary[:100]}...")
        
        results.append({
            'question': question,
            'model_answer': llm_output,
            'correct_answer': correct_answer,
            'score': result.overall,
            'passed': result.passed(),
            'faithfulness': result.faithfulness.score,
            'relevance': result.relevance.score,
            'toxicity': result.toxicity.score
        })
    
    # Calculate statistics
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    pass_rate = (passed / total) * 100 if total > 0 else 0
    avg_score = sum(r['score'] for r in results) / total if total > 0 else 0
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"truthfulqa_results_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'model_answer', 'correct_answer', 'score', 'passed', 'faithfulness', 'relevance', 'toxicity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Total Questions:    {total}")
    print(f"    Passed:          {passed} ({pass_rate:.1f}%)")
    print(f"    Failed:          {failed} ({100-pass_rate:.1f}%)")
    print(f"    Average Score:   {avg_score:.2f}/1.00")
    print("=" * 60)
    print(f"\n Detailed results saved to: {filename}")
    
    # Print individual results
    print("\n DETAILED RESULTS:")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"{i+1}. [{ 'PASS' if r['passed'] else 'FAIL' }] Score: {r['score']:.2f}")
        print(f"   Q: {r['question'][:60]}...")
        print(f"   A: {r['model_answer'][:60]}...")
        print()
    
    print("\n Next Steps:")
    print("   1. Open the CSV file to see detailed results")
    print("   2. Take screenshots of failed evaluations")
    print("   3. Share results on social media")
    print("\n Test complete!")


if __name__ == "__main__":
    main()