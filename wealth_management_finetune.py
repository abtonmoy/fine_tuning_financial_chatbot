#!/usr/bin/env python3
"""
Enhanced LoRA Fine-Tuning Script for Phi-3-mini-4k-instruct
Automatically detects and optimizes for multi-GPU or single GPU training
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

# Multi-GPU support
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    max_length: int = 1024  # Reduced from 2048 to save memory
    output_dir: str = "phi3-wealth-lora"
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training Configuration (will be adjusted based on GPU count)
    num_epochs: int = 3
    base_batch_size: int = 1  # Reduced from 2 to save memory
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
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

class GPUManager:
    """Manages GPU detection and configuration"""
    
    @staticmethod
    def check_flash_attention():
        """Check if Flash Attention 2 is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information and configuration"""
        if not torch.cuda.is_available():
            return {
                'available': False,
                'count': 0,
                'device_map': 'cpu',
                'use_ddp': False,
                'world_size': 1,
                'local_rank': 0
            }
        
        gpu_count = torch.cuda.device_count()
        
        # Check if we're in a distributed training environment
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        use_ddp = local_rank != -1 and world_size > 1
        
        if use_ddp:
            # We're in distributed training mode
            device_map = None  # Will be handled by DDP
        elif gpu_count > 1:
            # Multi-GPU single process (model parallelism)
            device_map = "auto"
        else:
            # Single GPU
            device_map = "auto"
        
        return {
            'available': True,
            'count': gpu_count,
            'device_map': device_map,
            'use_ddp': use_ddp,
            'world_size': world_size,
            'local_rank': local_rank
        }
    
    @staticmethod
    def setup_distributed():
        """Setup distributed training if needed"""
        gpu_info = GPUManager.get_gpu_info()
        
        if gpu_info['use_ddp']:
            # Initialize distributed training
            torch.cuda.set_device(gpu_info['local_rank'])
            dist.init_process_group(backend='nccl')
            logger.info(f"Initialized DDP on rank {gpu_info['local_rank']}/{gpu_info['world_size']}")
        
        return gpu_info
    
    @staticmethod
    def log_gpu_status(gpu_info):
        """Log current GPU configuration"""
        if not gpu_info['available']:
            logger.info("Training on CPU")
            return
        
        if gpu_info['use_ddp']:
            logger.info(f"Distributed training on {gpu_info['world_size']} GPUs")
            if gpu_info['local_rank'] == 0:
                for i in range(torch.cuda.device_count()):
                    logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        elif gpu_info['count'] > 1:
            logger.info(f"Multi-GPU training on {gpu_info['count']} GPUs")
            for i in range(gpu_info['count']):
                logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info(f"Single GPU training: {torch.cuda.get_device_name(0)}")

class WealthManagementTrainer:
    """Enhanced trainer for Phi-3 wealth management fine-tuning with auto GPU detection"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.datasets = None
        self.gpu_info = GPUManager.setup_distributed()
        
        # Adjust configuration based on GPU setup
        self._adjust_config_for_gpus()
        
        # Log GPU status
        GPUManager.log_gpu_status(self.gpu_info)
    
    def _adjust_config_for_gpus(self):
        """Adjust training configuration based on available GPUs"""
        gpu_count = self.gpu_info['count'] if self.gpu_info['available'] else 1
        
        if self.gpu_info['use_ddp']:
            # Distributed training: each process gets base_batch_size
            self.effective_batch_size = self.config.base_batch_size
            self.config.gradient_accumulation_steps = max(1, 8 // gpu_count)
        elif gpu_count > 1:
            # Multi-GPU single process: can use larger batch size
            self.effective_batch_size = min(self.config.base_batch_size * 2, 8)
            self.config.gradient_accumulation_steps = max(1, 8 // self.effective_batch_size)
        else:
            # Single GPU or CPU: use conservative settings
            self.effective_batch_size = self.config.base_batch_size
            self.config.gradient_accumulation_steps = max(1, 8 // self.effective_batch_size)
        
        logger.info(f"Training configuration:")
        logger.info(f"   Batch size per device: {self.effective_batch_size}")
        logger.info(f"   Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {self.effective_batch_size * self.config.gradient_accumulation_steps * max(1, gpu_count)}")
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with appropriate GPU configuration"""
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
        
        # Configure model loading based on GPU setup
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.gpu_info['available'] else torch.float32,
            "use_cache": False,  # Disable caching to save memory
        }
        
        if self.gpu_info['use_ddp']:
            # For DDP, load model on current device
            model_kwargs["device_map"] = f"cuda:{self.gpu_info['local_rank']}"
        elif self.gpu_info['available']:
            # For multi-GPU or single GPU
            model_kwargs["device_map"] = self.gpu_info['device_map']
            if self.gpu_info['count'] > 1:
                # Set memory limits for multi-GPU
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) * 0.8  # Use 80% of memory
                model_kwargs["max_memory"] = {i: f"{available_memory:.0f}GB" for i in range(self.gpu_info['count'])}
        else:
            # CPU training
            model_kwargs["device_map"] = "cpu"
        
        # Add attention implementation if GPU available
        if self.gpu_info['available']:
            if GPUManager.check_flash_attention():
                model_kwargs["attn_implementation"] = "flash_attention_2"
                if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
                    logger.info("Using Flash Attention 2 for optimized performance")
            else:
                # Fallback to eager attention if Flash Attention not available
                model_kwargs["attn_implementation"] = "eager"
                if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
                    logger.warning("Flash Attention 2 not available, using eager attention")
                    logger.info("For better performance, install with: pip install flash-attn --no-build-isolation")
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
        except RuntimeError as e:
            logger.error(f"Error loading model: {str(e)}")
            # If Flash Attention caused issues, retry without it
            if "flash_attention_2" in str(e):
                logger.warning("Retrying without Flash Attention due to compatibility issues")
                model_kwargs["attn_implementation"] = "eager"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )
            else:
                raise
        
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
        
        # # Enable gradient checkpointing to save memory
        # self.model.gradient_checkpointing_enable()
        # self.model.config.use_cache = False  # Disable cache for gradient checkpointing
        
        # Verify parameters are trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params} || All params: {total_params} || Trainable: {100 * trainable_params / total_params:.2f}%")
        
        # Only print on main process for DDP
        if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
            self.model.print_trainable_parameters()
        
    def load_and_prepare_data(self, data_path: str):
        """Load and prepare dataset with proper splits"""
        # Only log on main process for DDP
        if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
            logger.info(f"Loading data from: {data_path}")
        
        def load_jsonl(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        
        # Load data
        data = load_jsonl(data_path)
        
        if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
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
        
        if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
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
            desc="Tokenizing datasets" if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0 else None
        )
        
    def tokenize_function(self, examples):
        """Enhanced tokenization with proper chat formatting"""
        texts = []
        prompts = examples["prompt"] if isinstance(examples["prompt"], list) else [examples["prompt"]]
        responses = examples["response"] if isinstance(examples["response"], list) else [examples["response"]]
        
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
        
        # Setup training arguments based on GPU configuration
        training_args_dict = {
            "output_dir": self.config.output_dir,
            "per_device_train_batch_size": self.effective_batch_size,
            "per_device_eval_batch_size": self.effective_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_train_epochs": self.config.num_epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            
            # Logging and saving
            "logging_steps": 50,
            "eval_steps": 500,
            "save_steps": 500,
            "save_strategy": "steps",
            
            # Changed to match new Transformers API
            "eval_strategy": "steps",
            
            # Optimization
            "gradient_checkpointing": True,
            "dataloader_num_workers": 2 if self.gpu_info['available'] else 0,
            "dataloader_pin_memory": self.gpu_info['available'],
            "dataloader_drop_last": True,
            
            # Precision
            "bf16": self.gpu_info['available'],
            "fp16": False,
            
            # Miscellaneous
            "report_to": "none",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        
        # Add DDP-specific settings
        if self.gpu_info['use_ddp']:
            training_args_dict.update({
                "ddp_find_unused_parameters": True,  # Changed to True for LoRA compatibility
                "ddp_backend": "nccl",
                "local_rank": self.gpu_info['local_rank'],
            })
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if self.gpu_info['available'] else None
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
        if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
            logger.info("Starting training...")
        
        # Ensure model is in training mode
        self.model.train()
        
        train_result = trainer.train()
        
        # Save training metrics (only on main process)
        if not self.gpu_info['use_ddp'] or self.gpu_info['local_rank'] == 0:
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
        
        # Wait for all processes to finish
        if self.gpu_info['use_ddp']:
            dist.barrier()
        
        return trainer
    
    def test_inference(self, test_prompts: List[str] = None):
        """Test the fine-tuned model with sample prompts"""
        # Only run inference on main process
        if self.gpu_info['use_ddp'] and self.gpu_info['local_rank'] != 0:
            return
        
        if test_prompts is None:
            test_prompts = [
                "I want to invest $50,000 for retirement. What are my options?",
                "How can I diversify my investment portfolio?",
                "What's the difference between a Roth IRA and traditional IRA?"
            ]
        
        logger.info("Testing model inference...")
        
        # For inference, use single GPU or CPU
        device_map = "auto" if self.gpu_info['available'] else "cpu"
        torch_dtype = torch.bfloat16 if self.gpu_info['available'] else torch.float32
        
        # Load the fine-tuned model for inference
        model = AutoModelForCausalLM.from_pretrained(
            self.config.output_dir,
            device_map=device_map,
            torch_dtype=torch_dtype
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
            
            if self.gpu_info['available']:
                inputs = inputs.to(model.device)
            
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
    """Main training function with automatic GPU detection"""
    # Configuration that adapts to available hardware
    config = ModelConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        output_dir="phi3-wealth-lora",
        num_epochs=3,
        base_batch_size=1,  # Reduced to conserve memory
        learning_rate=2e-4,
        max_length=1024  # Reduced sequence length
    )
    
    # Initialize trainer (automatically detects and configures for available GPUs)
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
        
        # Test the model (only on main process)
        trainer.test_inference()
        
        # Final success message (only on main process)
        if not trainer.gpu_info['use_ddp'] or trainer.gpu_info['local_rank'] == 0:
            logger.info("Fine-tuning completed successfully!")
            logger.info(f"Model saved in: {config.output_dir}")
            logger.info("Your Phi-3 wealth management chatbot is ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Clean up distributed training
        if trainer.gpu_info['use_ddp']:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()