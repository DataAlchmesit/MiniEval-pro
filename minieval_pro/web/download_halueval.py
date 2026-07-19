from datasets import load_dataset
import json

print("📥 Downloading HaluEval dataset...")
print("This may take 1-2 minutes...")

# Choose which subset you want
dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

print(f"✅ Downloaded {len(dataset)} samples")
print(f"Sample: {dataset[0]}")

# Optionally save to JSON file
with open("halueval_data.json", "w") as f:
    # Convert to list of dictionaries for JSON serialization
    data_list = [dict(item) for item in dataset]
    json.dump(data_list, f, indent=2)

print("💾 Saved to halueval_data.json")