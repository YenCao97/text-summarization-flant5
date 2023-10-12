import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FlanT5RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = AutoModelForSeq2SeqLM.from_pretrained(config)
        self.config = model.config
        self.encoder = model.encoder
        self.decoder = model.decoder
        # Attach a randomly initialized linear head that outputs a scalar value on top
        self.v_head = nn.Linear(self.config.d_model, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.pad_id = self.tokenizer.pad_token_id

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        
        chosen_scores = []
        rejected_scores = []
        
        # Split the inputs and rewards into two parts, half is chosen and another half is rejected
        if len(input_ids.shape) == 2:
            batch_size = input_ids.shape[0] // 2
        chosen_summary = input_ids[:batch_size]
        rejected_summary = input_ids[batch_size:]
        chosen_rewards = rewards[:batch_size]
        rejected_rewards = rewards[batch_size:]

        # Compute pairwise loss. Only backprop on last value before padding
        loss = 0
         
        for i in range(batch_size):
            # Find the index of the first occurrence where input_ids of chosen_summary input_ids and rejected_summary are different
            divergence_idx = (chosen_summary[i] != rejected_summary[i]).nonzero()[0]
            
            # Check if there is any padding otherwise take length of sequence
            # Find the index of the first occurrence of the padding token of the chosen_summary
            chosen_idxs = (chosen_summary[i] == self.pad_id).nonzero()
            chosen_idx = chosen_idxs[0].item() if len(chosen_idxs) > 0 else chosen_summary.shape[1]
            # Find the index of the first occurrence of the padding token of the rejected_summary     
            rejected_idxs = (rejected_summary[i] == self.pad_id).nonzero()
            rejected_idx = rejected_idxs[0].item() if len(rejected_idxs) > 0 else rejected_summary.shape[1]
            
            end_idx = max(chosen_idx, rejected_idx)

            # Find the slice of reward which belongs to diverging input_ids
            chosen_truncated_reward = chosen_rewards[i][divergence_idx:end_idx]
            rejected_truncated_reward = rejected_rewards[i][divergence_idx:end_idx]

            # Append the last rewards to the list of end scores
            chosen_scores.append(chosen_truncated_reward[-1]) # reward at last token
            rejected_scores.append(rejected_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(chosen_truncated_reward - rejected_truncated_reward)).mean()

        loss = loss / batch_size
        chosen_scores = torch.stack(chosen_scores)
        # print("Chosen end scores: ", chosen_end_scores)
        rejected_scores = torch.stack(rejected_scores)
        # print("Rejected end scores: ", rejected_end_scores)
        
        return {
            "loss": loss,
            "chosen_scores": chosen_scores,
            "rejected_scores": rejected_scores,
        }
        