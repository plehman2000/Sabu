# using PyTorch 1.10

"""
Emotion Detection Module
"""

import os
import numpy as np
from transformers import BertModel
import transformers
import torch
from IPython.display import clear_output
from torch import nn
import onnx
import onnxruntime


#tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.3):
        #nn.Dropout(dropout),
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.layer1 = nn.Sequential( nn.Linear(768, 200), nn.ReLU(),nn.BatchNorm1d(200),nn.Linear(200, 200), nn.ReLU(),nn.Linear(200, 5), nn.Sigmoid() )

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        
        out = self.layer1(pooled_output)

        return out


DEVICE = torch.device('cpu')
MODELPATH = os.getcwd() + "\\DataModelling\\getEmotions\\emotionModel\\model.bin"
MODEL = BertClassifier()
ckpt = torch.load(open(MODELPATH, 'rb'), map_location=DEVICE)
MODEL.load_state_dict(ckpt)
print("\nModel Loaded!")

device = DEVICE
MODEL.to(DEVICE)
MODEL.eval()





def emotionInference(samples, model=MODEL, disp=True):
    model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    all_probs=[]
    for batch_index, sample in enumerate(samples):

        inputs = tokenizer.encode_plus(sample, None,add_special_tokens=True,max_length = 512,pad_to_max_length=True)

        ids = torch.unsqueeze((torch.tensor(inputs["input_ids"], dtype=torch.long)).to(device, dtype=torch.long), 0)
        token_type_ids = (torch.tensor(inputs["token_type_ids"], dtype=torch.long)).to(device, dtype=torch.long)
        mask = torch.unsqueeze((torch.tensor(inputs['attention_mask'], dtype=torch.long)).to(device, dtype=torch.long), 0)

        # Zero out any previously calculated gradients
        model.zero_grad()
        #Forward Pass (faux inference)
        logits = model(ids, mask)
        clear_output(wait=True)
        #nn.Softmax(dim=1)
        probabilities = (logits).detach().cpu().numpy()[0]
        all_probs.append(probabilities)
        
        
    #fear, sadness, joy, anger

    if disp:
        print(f"All Probabilities RAW: {all_probs}\n\n")
        for index, sample in enumerate(samples):
            print(f'Text: \"{sample}\"')
            #print(f'Logits: {logits[0].detach().cpu().numpy()}\n')
            ##order is Negative, Neutral, Positive
            percentages = all_probs[index]#may not work, delete code around if model actually softmaxes outputs
            #percentages = [100*num for num in all_probs[index]]
            #anger,disgust,fear,joy,sadness
            print(f'{100*percentages[0]:.2f}% anger, {100*percentages[1]:.2f}% disgust, {100*percentages[2]:.2f}% fear, {100*percentages[3]:.2f}% joy, {100*percentages[4]:.2f}% sadness')
    return all_probs






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





# # provide example input to the model and store output to assert correct export
# example_in = tokenize('Chancellor on brink of second bailout for banks')
# example_out = model(*example_in)

# # export the model
# torch.onnx.export(model, example_in, "bert_classifier.onnx", export_params=True,
#                   do_constant_folding=False, input_names=['input_ids', 'mask'],
#                   output_names=['output'])

# onnx_model = onnx.load("bert_classifier.onnx")
# onnx.checker.check_model(onnx_model)

# ort_session = onnxruntime.InferenceSession("bert_classifier.onnx")

# def to_numpy(tensor):
#     """Returns tensor as a NumPy ndarray"""
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# example_ort_ins = {ort_session.get_inputs()[0].name: to_numpy(example_in[0]),
#                    ort_session.get_inputs()[1].name: to_numpy(example_in[1])}
# example_ort_outs = ort_session.run(None, example_ort_ins)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(example_out), example_ort_outs[0], rtol=1e-03, atol=1e-05)
