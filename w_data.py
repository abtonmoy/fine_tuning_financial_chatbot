from datasets import load_dataset, Dataset
import json
from pathlib import Path
import pandas as pd
import requests
from io import StringIO

# Create output directory
output_dir = Path("phi3_wealth_data")
output_dir.mkdir(parents=True, exist_ok=True)

def save_jsonl(data, path):
    """Save data as JSONL format"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def safe_load_dataset(dataset_name, **kwargs):
    """Safely load dataset with error handling"""
    try:
        return load_dataset(dataset_name, **kwargs)
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {e}")
        return None

# -------------------------------
# ✅ Bitext - Wealth Management
# -------------------------------
print("Loading Bitext wealth management dataset...")
try:
    bitext = load_dataset("bitext/Bitext-wealth-management-llm-chatbot-training-dataset", split="train")
    print("✅ Bitext loaded successfully")
    print("Example Bitext row:", bitext[0])
    
    bitext_data = [
        {"prompt": ex["instruction"], "response": ex["response"]}
        for ex in bitext
        if ex.get("instruction") and ex.get("response")
    ]
    
    save_jsonl(bitext_data, output_dir / "bitext_wealth.jsonl")
    print(f"📁 Saved {len(bitext_data)} Bitext examples")
    
except Exception as e:
    print(f"❌ Failed to load Bitext: {e}")
    bitext_data = []

# -------------------------------
# ✅ FLUE - Financial PhraseBank
# -------------------------------
print("\nLoading Financial PhraseBank...")
try:
    flue = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
    print("✅ Financial PhraseBank loaded successfully")
    print("Example FLUE row:", flue[0])
    
    # Map sentiment labels to text
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    flue_data = [
        {
            "prompt": f"Analyze the sentiment of this financial statement: \"{ex['sentence']}\"",
            "response": f"The sentiment of this financial statement is {sentiment_map.get(ex['label'], 'unknown')}."
        }
        for ex in flue
        if ex.get('sentence')
    ]
    
    save_jsonl(flue_data, output_dir / "flue_sentiment.jsonl")
    print(f"📁 Saved {len(flue_data)} FLUE sentiment examples")
    
except Exception as e:
    print(f"❌ Failed to load Financial PhraseBank: {e}")
    flue_data = []

# -------------------------------
# 🔧 FinEval - Try alternative sources
# -------------------------------
print("\nTrying to load FinEval dataset...")
fineval_data = []

# Try different FinEval sources
fineval_urls = [
    "https://raw.githubusercontent.com/SUFE-AIFLM-Lab/FinEval/main/data/train/cfa_train.csv",
    "https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/raw/main/train/cfa_train.csv",
]

for url in fineval_urls:
    try:
        print(f"Trying URL: {url}")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            print("✅ FinEval loaded successfully")
            print(f"Dataset shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            
            # Convert to proper format
            fineval_data = []
            for _, row in df.iterrows():
                if pd.notnull(row.get("question")) and pd.notnull(row.get("answer")):
                    fineval_data.append({
                        "prompt": str(row["question"]),
                        "response": str(row["answer"])
                    })
            
            save_jsonl(fineval_data, output_dir / "fineval_qa.jsonl")
            print(f"📁 Saved {len(fineval_data)} FinEval examples")
            break
            
    except Exception as e:
        print(f"❌ Failed with URL {url}: {e}")
        continue

if not fineval_data:
    print("❌ Could not load FinEval from any source")

# -------------------------------
# 🔧 Alternative Financial Datasets
# -------------------------------
print("\nTrying alternative financial datasets...")

# Try FiQA dataset
try:
    fiqa = safe_load_dataset("fiqa", split="train")
    if fiqa:
        fiqa_data = [
            {
                "prompt": f"Answer this financial question: {ex['question']}",
                "response": ex["answer"]
            }
            for ex in fiqa
            if ex.get("question") and ex.get("answer")
        ]
        save_jsonl(fiqa_data, output_dir / "fiqa_qa.jsonl")
        print(f"📁 Saved {len(fiqa_data)} FiQA examples")
except:
    print("❌ FiQA dataset not available")

# Try financial news datasets
try:
    fin_news = safe_load_dataset("AdiOO7/financial-news-dataset", split="train")
    if fin_news:
        fin_news_data = [
            {
                "prompt": f"Summarize this financial news: {ex['text'][:500]}...",
                "response": f"This is a financial news article about {ex.get('label', 'financial markets')}."
            }
            for ex in fin_news[:1000]  # Limit to first 1000 examples
            if ex.get("text")
        ]
        save_jsonl(fin_news_data, output_dir / "financial_news.jsonl")
        print(f"📁 Saved {len(fin_news_data)} financial news examples")
except:
    print("❌ Financial news dataset not available")

# -------------------------------
# ✅ Combine All Available Datasets
# -------------------------------
print("\nCombining all available datasets...")
combined = []

# Collect all data
all_datasets = []
if bitext_data:
    all_datasets.append(("Bitext", bitext_data))
if flue_data:
    all_datasets.append(("FLUE", flue_data))
if fineval_data:
    all_datasets.append(("FinEval", fineval_data))

# Also check for any saved JSONL files
for file in output_dir.glob("*.jsonl"):
    if "combined" in file.name:
        continue
    
    print(f"Reading {file.name}...")
    with open(file, "r", encoding="utf-8") as f:
        file_data = []
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("prompt") and data.get("response"):
                    file_data.append(data)
            except json.JSONDecodeError:
                continue
        
        if file_data:
            combined.extend(file_data)

# Add data from successful loads
for name, data in all_datasets:
    combined.extend(data)
    print(f"✅ Added {len(data)} examples from {name}")

# Remove duplicates based on prompt
seen_prompts = set()
unique_combined = []
for item in combined:
    prompt_hash = hash(item["prompt"])
    if prompt_hash not in seen_prompts:
        seen_prompts.add(prompt_hash)
        unique_combined.append(item)

# Save combined dataset
if unique_combined:
    save_jsonl(unique_combined, output_dir / "combined_phi3_wealth_dataset.jsonl")
    print(f"\n✅ Combined dataset saved at: {output_dir/'combined_phi3_wealth_dataset.jsonl'}")
    print(f"📊 Total unique examples in combined set: {len(unique_combined)}")
    
    # Show dataset breakdown
    print("\n📈 Dataset Statistics:")
    for name, data in all_datasets:
        print(f"  - {name}: {len(data)} examples")
    
    print(f"\n🎯 Ready for Phi-3 fine-tuning!")
    print(f"Next steps:")
    print(f"1. Review the combined dataset: {output_dir/'combined_phi3_wealth_dataset.jsonl'}")
    print(f"2. Convert to Phi-3 format if needed")
    print(f"3. Start fine-tuning with your preferred framework")
    
else:
    print("❌ No data was successfully loaded. Please check your internet connection and dataset availability.")

print(f"\n📂 All output files saved in: {output_dir}")
print("Files created:")
for file in output_dir.glob("*.jsonl"):
    print(f"  - {file.name}")