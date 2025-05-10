# lora_finetuning.py

import json
import numpy as np
from pathlib import Path
from urllib import request
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import train, evaluate, TrainingArgs
from mlx_lm.tuner.datasets import CompletionsDataset
from mlx_lm.tuner import linear_to_lora_layers
import tqdm
import os

# Set environment variable for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configuration
SAVE_DIR = "hellaswag"
MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = Path("adapters")
NUM_TEST = 100

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

def make_dataset(ds, tokenizer):
    return CompletionsDataset(
        ds,
        tokenizer,
        prompt_key="instruction",
        completion_key="output",
        mask_prompt=False,
    )

def evaluate_model(model, tokenizer, test_set, num_test):
    num_correct = 0
    for prompt, completion, answer in tqdm.tqdm(test_set[:num_test]):    
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        tokens = tokenizer.apply_chat_template(messages, continue_final_message=True)
        response = generate(model, tokenizer, tokens, max_tokens=2)
        num_correct += (response==answer)
    return num_correct / num_test

# Training metrics class
class Metrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    
    def on_train_loss_report(self, info):
        self.train_losses.append((info["iteration"], info["train_loss"]))
    
    def on_val_loss_report(self, info):
        self.val_losses.append((info["iteration"], info["val_loss"]))

def main():
    # Load and prepare data
    print("Loading data...")
    train_set, test_set = load_json("train"), load_json("test")
    print(f"HellaSwag stats: {len(train_set)} training examples and {len(test_set)} test examples.")
    
    # Split training set into train and validation
    np.random.seed(43)
    perm = np.random.permutation(len(train_set))
    valid_size = int(0.1 * len(train_set))
    valid_set = [train_set[i] for i in perm[:valid_size]]
    train_set = [train_set[i] for i in perm[valid_size:]]
    
    # Setup LoRA configuration
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    
    lora_config = {
        "num_layers": 8,
        "lora_parameters": {
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        }
    }
    
    # Save LoRA config
    with open(ADAPTER_PATH / "adapter_config.json", "w") as fid:
        json.dump(lora_config, fid, indent=4)    
    
    # Setup training arguments
    training_args = TrainingArgs(
        adapter_file=ADAPTER_PATH / "adapters.safetensors",
        iters=200,
        steps_per_eval=50
    )
    
    # Load model and tokenizer
    print(f"Loading model {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)
    
    # Freeze model and convert to LoRA
    model.freeze()
    linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
    
    num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"Number of trainable parameters: {num_train_params}")
    
    # Prepare datasets
    train_set, valid_set = make_dataset(train_set, tokenizer), make_dataset(valid_set, tokenizer)
    
    # Setup training
    model.train()
    opt = optim.Adam(learning_rate=1e-5)
    metrics = Metrics()
    
    # Train model
    print("Starting training...")
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=metrics,
    )
    
    # Plot training metrics
    train_its, train_losses = zip(*metrics.train_losses)
    val_its, val_losses = zip(*metrics.val_losses)
    plt.figure()
    plt.plot(train_its, train_losses, '-o', label='Train')
    plt.plot(val_its, val_losses, '-o', label='Valid')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    # Prepare test set for evaluation
    test_set_eval = [(t["instruction"], *t["output"].rsplit(" ", maxsplit=1)) for t in test_set]
    
    # Evaluate the model
    print(f"Evaluating on {NUM_TEST} test examples...")
    test_acc = evaluate_model(model, tokenizer, test_set_eval, NUM_TEST)
    print(f"Approximate test accuracy: {test_acc:.3f}")
    
    # Optionally fuse the adapters (uncomment to enable)
    # print("Fusing adapters...")
    # os.system(f"mlx_lm.fuse --model {MODEL_PATH}")
    # 
    # # Evaluate fused model
    # model_fused, tokenizer_fused = load("fused_model")
    # test_acc_fused = evaluate_model(model_fused, tokenizer_fused, test_set_eval, NUM_TEST)
    # print(f"Fused model test accuracy: {test_acc_fused:.3f}")

if __name__ == "__main__":
    main()