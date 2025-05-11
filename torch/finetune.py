# pytorch_lora_finetuning.py

import json
import numpy as np
import torch
from pathlib import Path
from urllib import request
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Set environment variable for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configuration
SAVE_DIR = "hellaswag"
MODEL_PATH = "Qwen/Qwen3-4B"  # Using larger model for H100
ADAPTER_PATH = Path("adapters_pytorch")
NUM_TEST = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PERCENTAGE = 0.1  # Use 10% of the dataset for testing (set to 1.0 for full dataset)

# Helper functions
def download_and_save(save_dir):
    base_url = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/hellaswag/"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.json", "test.json"]:
        out_file = save_dir / name
        if not out_file.exists():
            request.urlretrieve(base_url + name, out_file)

def load_json(dataset):
    download_and_save(SAVE_DIR)
    with open(f"{SAVE_DIR}/{dataset}.json", "r") as fid:
        return json.load(fid)

# Custom dataset class
class HellaSwagDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["instruction"]
        completion = item["output"]

        # Create chat format
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]

        # Tokenize
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Create labels (same as input_ids for causal LM)
        labels = encodings["input_ids"].clone()

        # Mask the prompt tokens in labels (optional, similar to mask_prompt=False in MLX)
        prompt_tokens = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=True
        )
        labels[0, :len(prompt_tokens)] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def evaluate_model(model, tokenizer, test_set, num_test):
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for prompt, completion, answer in tqdm(test_set[:num_test]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]

            tokens = tokenizer.apply_chat_template(
                messages,
                continue_final_message=True,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model.generate(
                tokens,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(outputs[0][tokens.shape[1]:], skip_special_tokens=True)
            num_correct += (response.strip() == answer.strip())

    return num_correct / num_test

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    # Required method
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of the initialization of the Trainer."""
        return control

    # Optional methods that might be called by Trainer
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are available."""
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.val_losses.append((state.global_step, logs["eval_loss"]))
        return control

def main():
    # Load and prepare data
    print("Loading data...")
    train_set, test_set = load_json("train"), load_json("test")
    print(f"Original HellaSwag stats: {len(train_set)} training examples and {len(test_set)} test examples.")

    # Apply dataset percentage sampling
    if DATASET_PERCENTAGE < 1.0:
        np.random.seed(43)  # Keep seed for reproducibility
        train_size = int(len(train_set) * DATASET_PERCENTAGE)
        test_size = int(len(test_set) * DATASET_PERCENTAGE)

        train_indices = np.random.choice(len(train_set), train_size, replace=False)
        test_indices = np.random.choice(len(test_set), test_size, replace=False)

        train_set = [train_set[i] for i in train_indices]
        test_set = [test_set[i] for i in test_indices]

        print(f"Using {DATASET_PERCENTAGE*100}% of data: {len(train_set)} training examples and {len(test_set)} test examples.")

    # Split training set into train and validation
    np.random.seed(43)
    perm = np.random.permutation(len(train_set))
    valid_size = int(0.1 * len(train_set))
    valid_set = [train_set[i] for i in perm[:valid_size]]
    train_set = [train_set[i] for i in perm[valid_size:]]

    print(f"Final split: {len(train_set)} train, {len(valid_set)} validation examples.")

    # Setup LoRA configuration
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)

    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=20.0 * 8,  # scale * rank
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common for Qwen models
    )

    # Load model and tokenizer
    print(f"Loading model {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Add gradient checkpointing here
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for training with LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Prepare datasets
    train_dataset = HellaSwagDataset(train_set, tokenizer)
    valid_dataset = HellaSwagDataset(valid_set, tokenizer)

    # Adjust training steps based on dataset size
    max_steps = 200 if DATASET_PERCENTAGE == 1.0 else int(200 * DATASET_PERCENTAGE)
    eval_steps = 50 if DATASET_PERCENTAGE == 1.0 else max(10, int(50 * DATASET_PERCENTAGE))

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(ADAPTER_PATH),
        num_train_epochs=1,
        max_steps=200,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        logging_steps=10,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        eval_strategy="steps",  # Fixed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_torch",
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        remove_unused_columns=False,
        label_names=["labels"],  # Add this to suppress PEFT warning
        fp16=False,  # Use bf16 only, not fp16
        bf16=True,
        tf32=True,  # Enable tf32 for better stability
        dataloader_pin_memory=False,  # Can help with stability
    )

    # Setup metrics callback
    metrics_callback = MetricsCallback()

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,  # Changed from tokenizer to processing_class
        callbacks=[metrics_callback],
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Save the adapter
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    # Plot training metrics
    if metrics_callback.train_losses and metrics_callback.val_losses:
        train_its, train_losses = zip(*metrics_callback.train_losses)
        val_its, val_losses = zip(*metrics_callback.val_losses)
        plt.figure()
        plt.plot(train_its, train_losses, '-o', label='Train')
        plt.plot(val_its, val_losses, '-o', label='Valid')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('training_loss.png')
        plt.close()

    # Prepare test set for evaluation
    test_set_eval = [(t["instruction"], *t["output"].rsplit(" ", maxsplit=1)) for t in test_set]

    # Evaluate the model
    num_test_actual = min(NUM_TEST, len(test_set_eval))
    print(f"Evaluating on {num_test_actual} test examples...")
    test_acc = evaluate_model(model, tokenizer, test_set_eval, num_test_actual)
    print(f"Approximate test accuracy: {test_acc:.3f}")

    # Save final results
    results = {
        "test_accuracy": test_acc,
        "train_losses": metrics_callback.train_losses,
        "val_losses": metrics_callback.val_losses,
        "dataset_percentage": DATASET_PERCENTAGE,
        "total_train_examples": len(train_dataset),
        "total_val_examples": len(valid_dataset),
        "total_test_examples": len(test_set)
    }
    with open(ADAPTER_PATH / "results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
