# text-summarization-flant5

# Bachelor Thesis: Enhancing Text Capability of Language Models with Reinforcement Learning from Human Feedback

## Overview
This repository contains the code and resources for a bachelor thesis aimed at enhancing the text capability of language models using Reinforcement Learning from Human Feedback (RLHF). The project is based on the Flan-T5 language model and follows the RLHF approach outlined in the paper "Learning to Summarize with Human Feedback" by OpenAI.

## File Structure
The project files are organized into the following main stages of the development process:

1. Fine-tune Flan-T5: Contains the code and data necessary for fine-tuning the Flan-T5 language model.

2. Train Reward Model: Includes code and data related to training the reward model, a critical component in RLHF.

3. Train Policy with Reinforcement Learning: This section houses code and data for training the policy using reinforcement learning techniques.

## Data Sources
To implement this research, we use the following datasets and resources:

1. Reddit TL;DR Dataset: This dataset is available at [summarize_tldr](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr).

2. Human Preference Feedback Dataset: This dataset is available at [summarize_comparisons](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons).

3. Open-tool TRLX Framework: Accessible at [trlx](https://github.com/CarperAI/trlx/tree/main).

## Output Models
This research project has resulted in the following output models, which are available on Hugging-Face:

1. SFT Model: The SFT model can be accessed at [YenCao/sft-flan-t5](https://huggingface.co/YenCao/sft-flan-t5).

2. Reward Checkpoint: The reward checkpoint is available at [YenCao/reward-model-flan-t5](https://huggingface.co/YenCao/reward-model-flan-t5).

3. Policy Checkpoint: The policy checkpoint is available at [YenCao/policy-flan-t5](https://huggingface.co/YenCao/policy-flan-t5).
