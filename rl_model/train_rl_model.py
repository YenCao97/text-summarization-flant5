import torch
from datasets import load_dataset
from reward_model.rm_flant5 import FlanT5RewardModel
from transformers import AutoTokenizer
import json
import sys
import trlx
from trlx.data.default_configs import ILQLConfig, ModelConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, TrainConfig, TRLConfig


sft_path = "/home/cluster_home/cao/seq2seq_model/sft_model/sft_flan-t5/saved_model"
reward_path = "/home/cluster_home/cao/seq2seq_model/rm_model/rm_model_flant5/saved_model/pytorch_model.bin"

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        batch_size=8,
        epochs=100,
        total_steps=5000,
        checkpoint_interval=10000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="ilql_summarize_flant5",
    ),
    
    model=ModelConfig(model_path=sft_path, model_arch_type="seq2seq"),
    tokenizer=TokenizerConfig(tokenizer_path="google/flan-t5-large"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=5000, eta_min=1e-6)),
    method=ILQLConfig(name="ilqlconfig", tau=0.6, gamma=0.99, cql_scale=0.1, awac_scale=1, alpha=0.0001, beta=0, steps_for_target_q_sync=1, two_qs=True, gen_kwargs=dict(max_new_tokens=50, top_k=50, beta=[1, 2, 3], temperature=1.0)),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    #Load the tokenizer and the pre-trained reward model
    reward_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    reward_model = FlanT5RewardModel(sft_path)
    reward_model.load_state_dict(torch.load(reward_path), strict=False)
    reward_model.eval()

    def reward_function(samples):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [chosen for chosen in sub_samples]
            encodings_dict = reward_tokenizer(sub_samples, truncation=True, max_length=config.train.seq_length, padding="max_length", return_tensors="pt")
            input_ids = encodings_dict["input_ids"]
            attention_masks = encodings_dict["attention_mask"]
            input_ids = input_ids.repeat(2, 1)
            attention_masks = attention_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = reward_model(input_ids=input_ids, attention_mask=attention_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def preprocess_dataset(sample):
        sample["prompt_output"] = [
            [sample["prompt"] + " TL;DR:", sample["chosen"][7:]],
            [sample["prompt"] + " TL;DR:", sample["rejected"][7:]],
        ]
        sample["reward"] = [1, -1]
        return sample

    train_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")
    train_dataset = train_dataset.map(preprocess_dataset)

    prompts_outputs = sum(train_dataset["prompt_output"], [])
    rewards = sum(train_dataset["reward"], [])
    eval_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    eval_prompts = list(eval_dataset["prompt"])

    trlx.train(
        dataset=(prompts_outputs, rewards),
        metric_fn=lambda samples, **kwargs: {"rewards": reward_function(samples)},
        eval_prompts=eval_prompts,
        config=config,
    )


hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
main(hparams)
