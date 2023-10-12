from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import random
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")


# Load dataset, model and tokenizer of model
dataset_name = "CarperAI/openai_summarize_tldr"
model_name = "google/flan-t5-large"
dataset = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
random.seed(42)
# Unused columns for fine-tuning
remove_columns=["prompt", "label"]

# Get the maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
def get_max_length(dataset, column_key, tokenizer):
    tokenized_data = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x[column_key], truncation=True), 
        batched=True, 
        remove_columns=remove_columns
    )
    max_length = max([len(x) for x in tokenized_data["input_ids"]])
    return max_length

max_source_length = get_max_length(dataset, "prompt", tokenizer)
print(f"Max source sequence length: {max_source_length}")

max_target_length = get_max_length(dataset, "label", tokenizer)
print(f"Max target sequence length: {max_target_length}")

prefix = "summarize: "

def preprocess_function(examples, padding="max_length"):
    
    # Add prefix to the input for Flan-T5
    inputs = [prefix + example for example in examples["prompt"]]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=examples["label"], max_length=max_target_length, padding=padding, truncation=True)

    # Replace all tokenizer.pad_token_id in the labels by -100 because we want to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Load ROUGE metric
metric = evaluate.load("rouge")

# Function for post-processing text
def postprocess_text(predictions, labels):
    # Remove spaces at the beginning and at the end of texts
    predictions = [prediction.strip() for prediction in predictions]
    labels = [label.strip() for label in labels]

    # Operate texts at the level of sentences
    predictions = ["\n".join(sent_tokenize(prediction)) for prediction in predictions]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return predictions, labels


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions, decoded_labels = postprocess_text(decoded_predictions, decoded_labels)

    result = metric.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(prediction != tokenizer.pad_token_id) for prediction in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# Define data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    #pad_to_multiple_of=8
)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir="sft-flan-t5",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=False,
    learning_rate=5e-5,
    num_train_epochs=2,
    logging_dir="sft-flan-t5/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Create Seq2SeqTrainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
)

# Training model
print("Starting to train...")
trainer.train()
trainer.save_model("saved_model")
tokenizer.save_pretrained("saved_tokenizer")