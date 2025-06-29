#!/usr/bin/env python3
"""
Memory-Optimized Single-GPU LoRA Fine-Tuning Script for Phi-3-mini-4k-instruct
Optimized for NVIDIA RTX 4090 with 24GB VRAM
"""
import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import gc

# Core libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model and training parameters"""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_length: int = 512  # Further reduced to save memory
    output_dir: str = "phi3-wealth-lora"
    
    # LoRA Configuration - More conservative settings
    lora_r: int = 8  # Reduced from 16
    lora_alpha: int = 16  # Reduced from 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training Configuration - Memory optimized
    num_epochs: int = 3
    batch_size: int = 1  # Reduced to 1 for memory
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data split
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Quantization settings
    use_4bit: bool = True
    use_8bit: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            # Phi-3 specific target modules - reduced set for memory
            self.target_modules = [
                "q_proj", "v_proj",  # Only q and v projections to save memory
                "gate_proj", "down_proj"  # Reduced MLP targets
            ]

class WealthManagementTrainer:
    """Memory-optimized trainer for Phi-3 wealth management fine-tuning"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.datasets = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({total_memory:.1f}GB)")
        else:
            logger.info("Training on CPU - not recommended")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with memory optimization"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config for memory efficiency
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            quantization_config = None
        
        # Configure model loading with aggressive memory optimization
        model_kwargs = {
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        }
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA configuration with conservative settings
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable input gradients
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        
        # Clear cache again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Verify parameters are trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} || All params: {total_params:,} || Trainable: {100 * trainable_params / total_params:.2f}%")
        
        # Verify gradient setup
        grad_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        logger.info(f"Parameters requiring gradients: {len(grad_params)}")
        if len(grad_params) == 0:
            raise ValueError("No parameters require gradients! LoRA setup failed.")
        
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
    def load_and_prepare_data(self, data_path: str):
        """Load and prepare dataset with memory optimization"""
        logger.info(f"Loading data from: {data_path}")
        
        def load_jsonl(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        
        # Load data
        data = load_jsonl(data_path)
        logger.info(f"Loaded {len(data)} examples")
        
        # For memory optimization, limit dataset size during development
        if len(data) > 5000:  # Limit to 5k examples for memory efficiency
            logger.info(f"Limiting dataset to 5000 examples for memory optimization")
            data = data[:5000]
        
        # Create train/validation/test splits
        train_data, temp_data = train_test_split(
            data, 
            test_size=self.config.test_size + self.config.validation_size,
            random_state=42
        )
        
        val_size = self.config.validation_size / (self.config.test_size + self.config.validation_size)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=1-val_size,
            random_state=42
        )
        
        logger.info(f"Dataset splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create datasets
        self.datasets = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        # Tokenize datasets with smaller batch size
        self.datasets = self.datasets.map(
            self.tokenize_function,
            batched=True,
            batch_size=50,  # Reduced batch size for tokenization
            remove_columns=self.datasets['train'].column_names,
            desc="Tokenizing datasets"
        )
        
        # Clear memory after tokenization
        gc.collect()
        
    def tokenize_function(self, examples):
        """Memory-optimized tokenization function"""
        texts = []
        prompts = examples["prompt"]
        responses = examples["response"]
        
        for prompt, response in zip(prompts, responses):
            # Format for Phi-3 chat template
            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenize with reduced max length
        model_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors=None
        )
        
        # Set labels for language modeling
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        
        return model_inputs
    
    def train(self):
        """Execute the memory-optimized training process"""
        if self.model is None or self.datasets is None:
            raise ValueError("Model and datasets must be initialized before training")
        
        # Setup training arguments with aggressive memory optimization
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            
            # Logging and saving
            logging_steps=100,
            eval_steps=1000,
            save_steps=1000,
            save_strategy="steps",
            eval_strategy="steps",
            
            # Memory optimization
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,  # Disable to save memory
            dataloader_drop_last=True,
            max_grad_norm=1.0,
            
            # Precision
            bf16=True,
            fp16=False,
            
            # Memory management
            remove_unused_columns=False,
            save_total_limit=2,  # Only keep 2 checkpoints
            
            # Miscellaneous
            report_to="none",
            load_best_model_at_end=False,  # Disable to save memory
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['validation'],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Start training
        logger.info("Starting training...")
        try:
            train_result = trainer.train()
            
            # Save training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Out of memory during training: {e}")
            logger.info("Try reducing batch_size further or max_length")
            raise
        
        # Evaluate on test set (optional, skip if memory is tight)
        try:
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(eval_dataset=self.datasets['test'])
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
        except torch.cuda.OutOfMemoryError:
            logger.warning("Skipping test evaluation due to memory constraints")
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")
        
        return trainer
    
    def test_inference(self, test_prompts: List[str] = None):
        """Test the fine-tuned model with sample prompts"""
        if test_prompts is None:
            test_prompts = [
                "I want to invest $50,000 for retirement. What are my options?",
                "How can I diversify my investment portfolio?"
            ]
        
        logger.info("Testing model inference...")
        
        try:
            # Load the fine-tuned model for inference
            model = AutoModelForCausalLM.from_pretrained(
                self.config.output_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)
            
            for i, prompt in enumerate(test_prompts, 1):
                logger.info(f"\n--- Test {i} ---")
                logger.info(f"Prompt: {prompt}")
                
                # Format prompt
                chat = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Generate response with memory optimization
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=128,  # Reduced for memory
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                logger.info(f"Response: {response}")
                
        except Exception as e:
            logger.error(f"Inference test failed: {e}")

def main():
    """Main training function with memory optimization"""
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Configuration optimized for memory efficiency
    config = ModelConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        output_dir="phi3-wealth-lora",
        num_epochs=2,  # Reduced epochs
        batch_size=1,  # Minimum batch size
        learning_rate=2e-4,
        max_length=512,  # Reduced sequence length
        use_4bit=True,  # Enable 4-bit quantization
        lora_r=8,  # Conservative LoRA rank
        lora_alpha=16
    )
    
    # Initialize trainer
    trainer = WealthManagementTrainer(config)
    
    try:
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        # Load and prepare data
        data_path = "phi3_wealth_data/combined_phi3_wealth_dataset.jsonl"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        trainer.load_and_prepare_data(data_path)
        
        # Train the model
        trained_model = trainer.train()
        
        # Test the model (optional)
        try:
            trainer.test_inference()
        except Exception as e:
            logger.warning(f"Inference testing failed: {e}")
        
        # Final success message
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved in: {config.output_dir}")
        logger.info("Your Phi-3 wealth management chatbot is ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    # Set environment variables for memory optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    
    main()