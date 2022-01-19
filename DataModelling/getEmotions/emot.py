import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import transformers
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from tqdm import tqdm
import html
import regex
from IPython.display import clear_output
import scipy



MODELPATH = os.getcwd() + "/model/model.bin"
BATCH_SIZE = 8
DEVICE = 'cpu'

class DATALoader:
    def __init__(self, data, target, max_length):
        self.data = data
        self.target = target #make sure to convert the target into numerical values
        self.tokeniser = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.truncation=True
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())
        inputs = self.tokeniser.encode_plus(
            data, 
            None,
            add_special_tokens=True,
            max_length = self.max_length,
            pad_to_max_length=True
            
        )
        
        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        
        #added code to allow for inference function (needs to accept inpout without targets)
        if self.target == []:
            targets = []
        else:
            targets = torch.tensor(self.target[item], dtype=torch.float)


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': targets
        }


def getDataloaders(shuff=False, batch_size=4, val_fraction=1):
    dataDict = pickle.load(open('dframe.pkl', 'rb'))
    x = dataDict['samples'] 
    y = dataDict['labels']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=val_fraction, random_state=23)


    train_dataset = DATALoader(
    data=X_train,
    target=y_train,
    max_length=512
    )

    train_data_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    num_workers=0,
    shuffle=shuff
    )

    #print(train_dataset[0])
    index = int(len(X_test)*val_fraction)
    val_dataset = DATALoader(
    data=X_test[:index],
    target=y_test[:index],
    max_length=512
    )

    val_data_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=batch_size,
    num_workers=0,
    shuffle=shuff
    )
    
    
    debug_set = DATALoader(
    data=X_test[:10],
    target=y_test[:10],
    max_length=512
    )

    debug_loader = torch.utils.data.DataLoader(
    debug_set, 
    batch_size=1,
    num_workers=0,
    shuffle=shuff
    )
    
    return train_data_loader, val_data_loader, debug_loader

def Inference(samples, model, disp=True):
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
        for index, sample in enumerate(samples):
            print(f'Text: \"{sample}\"')
            #print(f'Logits: {logits[0].detach().cpu().numpy()}\n')
            ##order is Negative, Neutral, Positive
            percentages = all_probs[index]#may not work, delete code around if model actually softmaxes outputs
            #percentages = [100*num for num in all_probs[index]]
            print(f'{100*percentages[0]:.2f}% Fear, {100*percentages[1]:.2f}% Sadness, {100*percentages[2]:.2f}% Joy, {100*percentages[3]:.2f}% Anger')
    return all_probs


















def Train(data_loader, model=None, optimizer=None, device=None, scheduler=None, epochs=None, loss_fn =None, val=True):
    print("Beginning training...\n")
    ep_losses = []
    best_loss=1
    for epoch_i in range(epochs):
        print(f"Epoch: {epoch_i+1}\n")


        # Put the model into the training mode
        model.train()
        losses = []
        for batch_index, sample in tqdm(enumerate(data_loader), total=len(data_loader)):

            
            #Load batch variabes to GPU
            ids = (sample["ids"]).to(device, dtype=torch.long)
            token_type_ids = (sample["token_type_ids"]).to(device, dtype=torch.long)
            mask = (sample["mask"]).to(device, dtype=torch.long)
            targets = (sample["targets"]).to(device, dtype=torch.float) 

            # Zero out any previously calculated gradients
            model.zero_grad()
            #Forward Pass (faux inference)
            logits = model(ids, mask)
            
            clear_output(wait=True)

            loss = loss_fn(logits, targets)

            print(f"LOSS: {loss}")
        
            #Backpropagate the loss
            loss.backward()
        
            #Using the internally stored gradients, update weights/biases according to optimizer
            optimizer.step()
            scheduler.step()
            
            #Save most performant model

            

             # Calculate the average loss over the entire training data
            #avg_train_loss = total_loss / len(data_loader)
            #print(f'Average Loss: {avg_train_loss}')
            
            losses.append(loss)
        if val:
            losses = validate(model, valDataLoader)
            avg_loss = np.mean(losses)
            ep_losses.append(avg_loss)
            if avg_loss > best_loss:
                        pickle.dump({'epoch': epoch_i,
                                    'model_state_dict': model.state_dict()}, open(MODELPATH, 'wb'))
                        print("\nModel Saved!")
                        best_loss = loss
    print(f"Average Validation Loss: {avg_loss}")
    return losses, ep_losses

def validate(model, data_loader):
    return [] #TODO Change for trainings
    #model.load_state_dict(torch.load(MODELPATH))
    device = DEVICE
    model.eval()
    
    losses = []
    for batch_index, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        #Load batch variabes to GPU
        ids = (sample["ids"]).to(device, dtype=torch.long)
        token_type_ids = (sample["token_type_ids"]).to(device, dtype=torch.long)
        mask = (sample["mask"]).to(device, dtype=torch.long)
        targets = (sample["targets"]).to(device, dtype=torch.float) 

        # Zero out any previously calculated gradients
        model.zero_grad()
        #Forward Pass (faux inference)
        logits = model(ids, mask)

        clear_output(wait=True)

        loss = loss_fn(logits, targets)

        print(f"LOSS: {loss}")

        losses.append(loss.detach().cpu().numpy())
    return losses


from transformers import BertModel
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.layer1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 100), nn.ReLU(),nn.BatchNorm1d(100),nn.Linear(100, 4), nn.ReLU() )

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        
        out = self.layer1(pooled_output)

        return out




dataDict = pickle.load(open('dframe.pkl', 'rb'))
#fear, sadness, joy, anger
x = dataDict['samples'] 
y = dataDict['labels']

print(f'Data Length: {len(x)}\n')


import time
import pickle
from transformers import AdamW, get_linear_schedule_with_warmup



BATCH_SIZE=8

trainDataLoader, valDataLoader, debugLoader = getDataloaders(False, batch_size=BATCH_SIZE, val_fraction = 0.1)#change back todo

print("LENGTH: ",len(trainDataLoader))
useSavedModel = False
model = BertClassifier()
if useSavedModel:
        print("Loading Saved Model...")
        checkpoint = pickle.load(open(MODELPATH, 'rb'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\nModel Loaded!")
else:
    l_r =1e-4
    # l_r = 5e-5
    #eps = 1e-8
    eps = 1e-8
    epochs=1

    loss_fn = nn.MSELoss()

    devName = DEVICE #change back when putting on gpu



    total_steps = len(trainDataLoader) * epochs
    optimizer = AdamW(model.parameters(), lr=l_r, eps=eps)
    device = torch.device(devName)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) #0 is default for warmup steps

    torch.cuda.empty_cache()
    model.to(device)


    losses, ep_losses = Train(trainDataLoader, model, optimizer, device, scheduler, epochs, loss_fn)
    torch.save(model.state_dict(), MODELPATH)

samples = ["i hate dogs!", "I love pretty cats!",  "There will be heavy rainstorms tomorrow", 'fuck i am angry']
probs = Inference(samples, model)
