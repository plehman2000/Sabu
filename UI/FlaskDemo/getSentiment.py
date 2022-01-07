# using pyorch 1.10


import os
import numpy as np
from transformers import BertModel
import transformers
import torch
import torch.nn as nn


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.layer1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 100), nn.ReLU(), nn.Linear(100, 3), nn.ReLU())

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask)

        out = self.layer1(pooled_output)

        return out


MODELPATH = os.getcwd() + '\\'
stat_dict = torch.load(MODELPATH + 'model.pkl', map_location='cpu')  # device was cuda:0
model = BertClassifier()
print(stat_dict["bert.embeddings.position_ids"])
stat_dict.pop("bert.embeddings.position_ids", None)
model.load_state_dict(stat_dict)
model.to('cpu')
device = torch.device('cpu')


def Inference(samples, disp=True):
    with torch.no_grad():
        model.eval()
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        all_probs = []
        for batch_index, sample in enumerate(samples):
            inputs = tokenizer.encode_plus(sample, None, add_special_tokens=True, max_length=512,
                                           pad_to_max_length=True)

            ids = torch.unsqueeze((torch.tensor(inputs["input_ids"], dtype=torch.long)).to(device, dtype=torch.long), 0)
            token_type_ids = (torch.tensor(inputs["token_type_ids"], dtype=torch.long)).to(device, dtype=torch.long)
            mask = torch.unsqueeze(
                (torch.tensor(inputs['attention_mask'], dtype=torch.long)).to(device, dtype=torch.long), 0)

            # Zero out any previously calculated gradients
            model.zero_grad()
            # Forward Pass (faux inference)
            logits = model(ids, mask)

            probabilities = (logits).detach().cpu().numpy()[0]
            all_probs.append(probabilities)

        if disp:
            for index, sample in enumerate(samples):
                print(f'Text: \"{sample}\"')
                ##order is Negative, Neutral, Positive
                percentages = all_probs[index]

                print(
                    f'{100 * percentages[0]:.2f}% Negative, {100 * percentages[1]:.2f}% Neutral, {100 * percentages[2]:.2f}% Positive')
    return all_probs


def getSentimentAverage(samples, disp=False):
    probs = Inference(samples, False)
    neg_avg = np.mean([row[0] for row in probs])
    neu_avg = np.mean([row[1] for row in probs])
    pos_avg = np.mean([row[2] for row in probs])
    if disp:
        print(f'{100 * neg_avg:.2f}% Negative, {100 * neu_avg:.2f}% Neutral, {100 * pos_avg:.2f}% Positive')
    return [pos_avg, neu_avg, neg_avg]