# using PyTorch 1.10

"""
Sentiment analysis module
"""

import os
import numpy as np
from transformers import BertModel
import transformers
import torch
from torch import nn
import onnx
import onnxruntime

MODELPATH = os.getcwd() + '\\DataModelling\\getSentiment\\sentimentModel\\' #needs to be changed when deployed
stat_dict = torch.load(MODELPATH + 'model.pkl', map_location='cpu') # device was cuda:0
device = torch.device('cpu')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

def tokenize(sample):
    """Tokenizes sample and returns input IDs and attention mask"""
    inputs = tokenizer.encode_plus(sample, None, add_special_tokens=True, max_length=512,
                                   padding='max_length', truncation=True)

    ids = torch.unsqueeze(
        torch.tensor(inputs["input_ids"], dtype=torch.long).to(device, dtype=torch.long),
        0)

    mask = torch.unsqueeze(
        torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device, dtype=torch.long),
        0)

    return ids, mask

class BertClassifier(nn.Module):
    """BERT classifier"""
    def __init__(self, dropout=0.5):

        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.layer1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 100),
                                    nn.ReLU(), nn.Linear(100, 3), nn.ReLU())

    def forward(self, input_id, mask):
        """Feed input to BERT and classifier"""
        _, pooled_output = self.bert(input_ids=input_id,
                                     attention_mask=mask,
                                     return_dict=False)

        out = self.layer1(pooled_output)

        return out

model = BertClassifier()
model.load_state_dict(stat_dict)
model.to('cpu')
model.eval()
# provide example input to the model and store output to assert correct export
example_in = tokenize('Chancellor on brink of second bailout for banks')
example_out = model(*example_in)

# export the model
torch.onnx.export(model, example_in, "bert_classifier.onnx", export_params=True,
                  do_constant_folding=False, input_names=['input_ids', 'mask'],
                  output_names=['output'])

onnx_model = onnx.load("bert_classifier.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("bert_classifier.onnx")

def to_numpy(tensor):
    """Returns tensor as a NumPy ndarray"""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
example_ort_ins = {ort_session.get_inputs()[0].name: to_numpy(example_in[0]),
                   ort_session.get_inputs()[1].name: to_numpy(example_in[1])}
example_ort_outs = ort_session.run(None, example_ort_ins)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(example_out), example_ort_outs[0], rtol=1e-03, atol=1e-05)

model.train()

def sentimentInference(samples, disp=True, optimize=False):
    """Perform predictive inference"""
    all_probs = []
    if not optimize:
        with torch.no_grad():
            model.eval()
            for _, sample in enumerate(samples):

                # token_type_ids = (
                # torch.tensor(inputs["token_type_ids"], dtype=torch.long)
                #              .to(device, dtype=torch.long))

                # zero out any previously calculated gradients
                model.zero_grad()
                # forward pass (faux inference)
                logits = model(*tokenize(sample))

                probabilities = logits.detach().cpu().numpy()[0]
                all_probs.append(probabilities)
    else:
        for _, sample in enumerate(samples):
            ids, mask = tokenize(sample)
            ort_ins = {ort_session.get_inputs()[0].name: to_numpy(ids),
                       ort_session.get_inputs()[1].name: to_numpy(mask)}
            ort_outs = ort_session.run(None, ort_ins)
            probabilities = ort_outs[0][0]
            all_probs.append(probabilities)
    if disp:
        for index, sample in enumerate(samples):
            print(f'Text: \"{sample}\"')
            # order is Negative, Neutral, Positive
            percentages = all_probs[index]

            print(
                f'{100*percentages[0]:.2f}% Negative, '
                f'{100*percentages[1]:.2f}% Neutral, '
                f'{100*percentages[2]:.2f}% Positive')
    return all_probs


def getSentimentAverage(samples, disp=False, optimize=False):
    """Get average sentiment for each polarity"""
    probs = Inference(samples, False, optimize)
    avgs = [np.mean([row[i] for row in probs]) for i in range(3)]
    reciprocal_sum = 1 / sum(avgs)
    avgs = [avg * reciprocal_sum for avg in avgs]
    if disp:
        print(
            f'{100*avgs[0]:.2f}% Negative, {100*avgs[1]:.2f}% Neutral, {100*avgs[2]:.2f}% Positive')
    return avgs
