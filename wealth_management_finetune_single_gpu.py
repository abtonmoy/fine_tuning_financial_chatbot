#!/usr/bin/env python3
"""
Single-GPU LoRA Fine-Tuning Script for Phi-3-mini-4k-instruct
Optimized for NVIDIA RTX 4090
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

# Core libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model and training parameters"""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_length: int = 1024  # Reduced to save memory
    output_dir: str = "phi3-wealth-lora"
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training Configuration
    num_epochs: int = 3
    batch_size: int = 4  # Optimized for single GPU
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data split
    test_size: float = 0.2
    validation_size: float = 0.1
    
    def __post_init__(self):
        if self.target_modules is None:
            # Phi-3 specific target modules
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

class WealthManagementTrainer:
    """Trainer for Phi-3 wealth management fine-tuning with single GPU"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.datasets = None
        
        # Log GPU status
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
        else:
            logger.info("Training on CPU - not recommended")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer for single GPU"""
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
        
        # Configure model loading
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "use_cache": False,  # Disable caching to save memory
        }
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Setup LoRA configuration
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
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        # Verify parameters are trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} || All params: {total_params:,} || Trainable: {100 * trainable_params / total_params:.2f}%")
        
    def load_and_prepare_data(self, data_path: str):
        """Load and prepare dataset with proper splits"""
        logger.info(f"Loading data from: {data_path}")
        
        def load_jsonl(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        
        # Load data
        data = load_jsonl(data_path)
        logger.info(f"Loaded {len(data)} examples")
        
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
        
        # Tokenize datasets in batches to reduce memory usage
        self.datasets = self.datasets.map(
            self.tokenize_function,
            batched=True,
            batch_size=100,  # Process 100 examples at a time
            remove_columns=self.datasets['train'].column_names,
            desc="Tokenizing datasets"
        )
        
    def tokenize_function(self, examples):
        """Enhanced tokenization with proper chat formatting"""
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
        
        # Tokenize
        model_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=self.config.max_length,
            return_tensors=None
        )
        
        # Set labels for language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        """Compute perplexity for evaluation"""
        predictions, labels = eval_pred
        
        # Flatten predictions and labels
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss manually
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))
        perplexity = torch.exp(loss)
        
        return {"perplexity": perplexity.item()}
    
    def train(self):
        """Execute the training process"""
        if self.model is None or self.datasets is None:
            raise ValueError("Model and datasets must be initialized before training")
        
        # Setup training arguments
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
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
            save_strategy="steps",
            evaluation_strategy="steps",
            
            # Optimization
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            
            # Precision
            bf16=True,
            fp16=False,
            
            # Miscellaneous
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['validation'],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=self.datasets['test'])
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        trainer.save_state()
        
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")
        
        return trainer
    
    def test_inference(self, test_prompts: List[str] = None):
        """Test the fine-tuned model with sample prompts"""
        if test_prompts is None:
            test_prompts = [
                "I want to invest $50,000 for retirement. What are my options?",
                "How can I diversify my investment portfolio?",
                "What's the difference between a Roth IRA and traditional IRA?"
            ]
        
        logger.info("Testing model inference...")
        
        # Load the fine-tuned model for inference
        model = AutoModelForCausalLM.from_pretrained(
            self.config.output_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16
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
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            logger.info(f"Response: {response}")

def main():
    """Main training function"""
    # Configuration optimized for single GPU
    config = ModelConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        output_dir="phi3-wealth-lora",
        num_epochs=3,
        batch_size=4,  # Optimized for RTX 4090
        learning_rate=2e-4,
        max_length=1024
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
        
        # Test the model
        trainer.test_inference()
        
        # Final success message
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved in: {config.output_dir}")
        logger.info("Your Phi-3 wealth management chatbot is ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set environment variables to prevent tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run with memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    main()