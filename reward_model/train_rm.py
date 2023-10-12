from attr import dataclass
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from rm_flant5 import FlanT5RewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import random

# def set_seed(seed_val=42):
#   random.seed(seed_val)
#   np.random.seed(seed_val)
#   torch.manual_seed(seed_val)
#   torch.cuda.manual_seed_all(seed_val)

# Load dataset
dataset_name = "CarperAI/openai_summarize_comparisons"
training_dataset = load_dataset(dataset_name, split="train").shuffle(seed=42)
eval_dataset = load_dataset(dataset_name, split="valid2").shuffle(seed=42)
baseline = "google/flan-t5-large"

# Initialize the reward model from the SFT model
reward_model = FlanT5RewardModel("YenCao/sft-flan-t5")
print("Loaded model")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(baseline)


# Pre-prrocess the dataset that concentrates a post/prompt to its corresponding chosen and rejected summary
def preprocess_dataset(dataset):
    comparison_pairs = []
    for sample in dataset:
        post = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        # Skip if chosen and rejected summaries are the same or too short
        if chosen_summary == rejected_summary or len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        comparison_pair = {
            "chosen": f"{post}\n{chosen_summary}",
            "rejected": f"{post}\n{rejected_summary}"
        }
        comparison_pairs.append(comparison_pair)

    return comparison_pairs

train_pairs = preprocess_dataset(training_dataset)
# print(train_pairs[0])
eval_pairs = preprocess_dataset(eval_dataset)
# print(eval_pairs[0])

# Tokenize the dataset
class TokenizedDataset(Dataset):
    def __init__(self, comparison_pairs, tokenizer):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for comparison_pair in comparison_pairs:
            chosen = comparison_pair["chosen"]
            rejected = comparison_pair["rejected"]
            tokenized_chosen = tokenizer(chosen, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            tokenized_rejected = tokenizer(rejected, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            if not torch.all(torch.eq(tokenized_chosen["input_ids"], tokenized_rejected["input_ids"])).item():
                self.chosen_input_ids.append(tokenized_chosen["input_ids"])
                self.chosen_attn_masks.append(tokenized_chosen["attention_mask"])
                self.rejected_input_ids.append(tokenized_rejected["input_ids"])
                self.rejected_attn_masks.append(tokenized_rejected["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        # return {
        #     "chosen_input_ids": self.chosen_input_ids[idx],
        #     "chosen_attn_masks": self.chosen_attn_masks[idx],
        #     "rejected_input_ids": self.rejected_input_ids[idx],
        #     "rejected_attn_masks": self.rejected_attn_masks[idx],
        # }
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )
        
# Collate data for training model        
class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch
    
train_dataset = TokenizedDataset(train_pairs, tokenizer)
# print(train_dataset[0])
eval_dataset = TokenizedDataset(eval_pairs, tokenizer)
# print(eval_dataset[0])

# Create the collator to gather batches of pairwise comparisons
data_collator = DataCollatorReward()  

def compute_metrics(eval_preds):
    chosen_scores = eval_preds.predictions[0]  # chosen scores
    rejected_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_scores > rejected_scores) / len(rejected_scores)
    result["accuracy"] = acc

    return result

random.seed(42)
training_args = TrainingArguments(
    output_dir="rw-model",
    num_train_epochs=1,
    logging_steps=10,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    eval_steps=1000, 
    save_steps=1000,
    warmup_steps=100,
    #logging_dir="./logs",
    fp16=False,
    learning_rate=5e-5,
    save_total_limit=1,
)

# Define Trainer instance
trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("Starting to train ...")
trainer.train()
trainer.save_model("saved_model")
tokenizer.save_pretrained("saved_tokenizer")
