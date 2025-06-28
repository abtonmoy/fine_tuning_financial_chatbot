import json
import random
from pathlib import Path
from collections import Counter

# Load and inspect the combined dataset
data_path = Path("phi3_wealth_data/combined_phi3_wealth_dataset.jsonl")

def load_jsonl(filepath):
    """Load JSONL data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

def analyze_dataset(data):
    """Analyze dataset statistics"""
    print("🔍 DATASET ANALYSIS")
    print("=" * 50)
    print(f"📊 Total examples: {len(data)}")
    
    # Analyze prompt lengths
    prompt_lengths = [len(ex['prompt']) for ex in data]
    response_lengths = [len(ex['response']) for ex in data]
    
    print(f"\n📝 PROMPT STATISTICS:")
    print(f"  Average length: {sum(prompt_lengths)/len(prompt_lengths):.1f} chars")
    print(f"  Min length: {min(prompt_lengths)} chars")
    print(f"  Max length: {max(prompt_lengths)} chars")
    
    print(f"\n💬 RESPONSE STATISTICS:")
    print(f"  Average length: {sum(response_lengths)/len(response_lengths):.1f} chars")
    print(f"  Min length: {min(response_lengths)} chars")
    print(f"  Max length: {max(response_lengths)} chars")
    
    # Show sample data
    print(f"\n🎯 SAMPLE EXAMPLES:")
    print("=" * 50)
    
    # Show a few random examples
    samples = random.sample(data, min(3, len(data)))
    for i, ex in enumerate(samples, 1):
        print(f"\n📋 Example {i}:")
        print(f"PROMPT: {ex['prompt'][:200]}{'...' if len(ex['prompt']) > 200 else ''}")
        print(f"RESPONSE: {ex['response'][:200]}{'...' if len(ex['response']) > 200 else ''}")
        print("-" * 40)

def convert_to_phi3_format(data, output_path):
    """Convert to Phi-3 chat format"""
    print("\n🔄 CONVERTING TO PHI-3 FORMAT")
    print("=" * 50)
    
    phi3_data = []
    for ex in data:
        # Phi-3 uses ChatML format
        phi3_example = {
            "messages": [
                {
                    "role": "user",
                    "content": ex["prompt"]
                },
                {
                    "role": "assistant", 
                    "content": ex["response"]
                }
            ]
        }
        phi3_data.append(phi3_example)
    
    # Save in Phi-3 format
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in phi3_data:
            json.dump(ex, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Phi-3 format saved to: {output_path}")
    print(f"📊 Converted {len(phi3_data)} examples")
    
    return phi3_data

def create_training_splits(data, train_ratio=0.8):
    """Split data into train/validation sets"""
    print(f"\n✂️ CREATING TRAIN/VALIDATION SPLITS")
    print("=" * 50)
    
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"📚 Training examples: {len(train_data)}")
    print(f"🧪 Validation examples: {len(val_data)}")
    
    return train_data, val_data

def save_phi3_splits(train_data, val_data, output_dir):
    """Save train/val splits in Phi-3 format"""
    output_dir = Path(output_dir)
    
    # Save training set
    train_path = output_dir / "phi3_train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train_data:
            json.dump(ex, f, ensure_ascii=False)
            f.write('\n')
    
    # Save validation set
    val_path = output_dir / "phi3_val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val_data:
            json.dump(ex, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Training data saved to: {train_path}")
    print(f"✅ Validation data saved to: {val_path}")

def create_phi3_config():
    """Create a sample fine-tuning config for Phi-3"""
    config = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "dataset": {
            "train_file": "phi3_train.jsonl",
            "validation_file": "phi3_val.jsonl",
            "max_length": 2048
        },
        "training": {
            "learning_rate": 5e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 50
        },
        "optimization": {
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "lora": {
            "enabled": True,
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    }
    
    config_path = Path("phi3_wealth_data/phi3_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️ Phi-3 config saved to: {config_path}")
    return config

# Main execution
if __name__ == "__main__":
    print("🚀 PHI-3 WEALTH MANAGEMENT DATASET PROCESSOR")
    print("=" * 60)
    
    # Load data
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    print(f"📂 Loading data from: {data_path}")
    data = load_jsonl(data_path)
    
    if not data:
        print("❌ No data loaded!")
        exit(1)
    
    # Analyze dataset
    analyze_dataset(data)
    
    # Convert to Phi-3 format
    phi3_path = Path("phi3_wealth_data/phi3_format.jsonl")
    phi3_data = convert_to_phi3_format(data, phi3_path)
    
    # Create train/val splits
    train_data, val_data = create_training_splits(phi3_data)
    
    # Save splits
    save_phi3_splits(train_data, val_data, "phi3_wealth_data")
    
    # Create config
    config = create_phi3_config()
    
    print(f"\n🎯 READY FOR PHI-3 FINE-TUNING!")
    print("=" * 50)
    print("📁 Files created:")
    print("  ├── phi3_train.jsonl          (Training data)")
    print("  ├── phi3_val.jsonl            (Validation data)")
    print("  ├── phi3_format.jsonl         (Full dataset)")
    print("  └── phi3_config.json          (Training config)")
    
    print(f"\n🔥 NEXT STEPS:")
    print("1. Review the data splits and config")
    print("2. Choose your fine-tuning framework:")
    print("   • Hugging Face Transformers + PEFT")
    print("   • Unsloth (fastest option)")
    print("   • LLaMA-Factory") 
    print("   • MLX (for Mac)")
    print("3. Start fine-tuning with the generated files!")
    
    print(f"\n💡 SAMPLE TRAINING COMMAND (Transformers):")
    print("python -m transformers.trainer \\")
    print("    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \\")
    print("    --train_file phi3_train.jsonl \\")
    print("    --validation_file phi3_val.jsonl \\")
    print("    --output_dir ./phi3-wealth-model \\")
    print("    --num_train_epochs 3 \\")
    print("    --per_device_train_batch_size 4 \\")
    print("    --gradient_accumulation_steps 4 \\")
    print("    --learning_rate 5e-5 \\")
    print("    --save_steps 500 \\")
    print("    --eval_steps 500")