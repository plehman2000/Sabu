import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.utils import shuffle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from tqdm import tqdm
import html
import regex


from inference import Inference

HASHTAG_CHARS = r"\p{L}\p{M}\p{Nd}_\u200c\u200d\ua67e\u05be\u05f3\u05f4\uff5e\u301c\u309b\u309c\u30a0\u30fb\u3003\u0f0b\u0f0c\u00b7"
MODELPATH = os.getcwd() + "\\checkpoints\\model.bin"


def process_tweet(tweet):
    """Remove all URLs (e.g. www.xyz.com), hash tags (e.g. #topic), targets (@username)"""
    
    tweet = regex.sub(r"https?://t\.co/[a-zA-Z0-9]+",
                                "", tweet)

    tweet = regex.sub(r"(?:([^\w!#$%&*@＠]|^)|(?:^|[^\w+~.-])(?:rt|rT|Rt|RT):?)[@＠](\w{1,20}(?:/[a-zA-Z][\w-]{0,24})?)",
                                r"\1\2", tweet)

    tweet  = regex.sub(r"(^|\ufe0e|\ufe0f|[^&" +
                                HASHTAG_CHARS +
                                r"])[#＃]((?!\ufe0f|\u20e3)[" +
                                HASHTAG_CHARS +
                                r"]*[\p{L}\p{M}][" +
                                HASHTAG_CHARS +
                                r"]*)",
                                r"\1\2", tweet)

    tweet = regex.sub(r"\n+",
                                "\n", tweet)

    tweet = regex.sub(r"\s+",
                                " ", tweet).strip()

    tweet = html.unescape(tweet)

    return tweet

class BERTPrediction(nn.Module):
    def __init__ (self):
        super(BERTPrediction, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        self.bert_drop = nn.Dropout(0.4)
        self.out = nn.Linear(768, 3)
        
    def forward(self, ids, masks, token_type_ids):
        _, pooledOut = self.bert(ids, attention_mask = masks,
                                token_type_ids=token_type_ids, return_dict=False)
        bertOut = self.bert_drop(pooledOut)
        output = self.out(bertOut)
        
        return output
    
class DATALoader:
    def __init__(self, data, target, max_length):
        self.data = data
        self.target = target #make sure to convert the target into numerical values
        self.tokeniser = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        
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
    
def loss_fn(output, targets):
    return nn.BCEWithLogitsLoss()(output, targets)


def train_func(data_loader, model, optimizer, device, scheduler, batchFraction=1):
    model.to(device)
    
    if batchFraction==0: 
        batchLimit=1 
    else: 
         batchLimit = int((len(data_loader)/(data_loader.batch_size) * batchFraction)) * (data_loader.batch_size)
  #  print(f"\nNUMBATCHES: {len(data_loader)}")
   # print(f"\n\nBATCH FRACTION:  {batchLimit}\n")
    

    
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        if bi > batchLimit:
            print("\nTRAINING RUN COMPLETE")
            break
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
#         print(f'D: {d}')
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(
            ids=ids,
            masks = mask,
            token_type_ids = token_type_ids
        )
        
        #print(f'Outputs: {output}')
       # print(f'Targets: {targets.view(-1,1)}')
        loss = loss_fn(output, targets)
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
def eval_func(data_loader, model, device, batchFraction):
    model.eval()
    
    if batchFraction==0: 
        batchLimit=1 
    else: 
         batchLimit = int((len(data_loader)/(data_loader.batch_size) * batchFraction)) * (data_loader.batch_size)
         
    fin_targets = []
    fin_outputs = []
    
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            if bi > batchLimit:
                print("\nEVALUATION COMPLETE")
                break  
        
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


            output = model(
                ids=ids,
                masks = mask,
                token_type_ids = token_type_ids
            )
        
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
            
        return fin_outputs, fin_targets
    
def run(devName, useSavedModel=False, epochs=1, batchFraction=1):
    dataset = pd.read_csv(os.getcwd() + "/data/tweet_text_with_scores_f10k.csv")
    dataset.drop('id',inplace=True, axis=1)
    # dataset.head()
    dataset = shuffle(dataset,random_state=42)
    
    tqdm.pandas()
    dataset['text'] = dataset['text'].progress_apply(lambda x: process_tweet(x))
    
    x = dataset['text'].values
    #Rounding the labels to 4 decimal places, so readouts can be of the form "99.23% Positive"
    precision = 4
    
    negList = []
    for label in dataset['NEG'].values: negList.append(round(label, precision))

    neutralList = []
    for label in dataset['NEU'].values: neutralList.append(round(label, precision))
    
    posList =  []
    for label in dataset['POS'].values : posList.append(round(label, precision))




    y = np.array(tuple(zip( negList, neutralList, posList)))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=23)


    train_dataset = DATALoader(
        data=X_train,
        target=y_train,
        max_length=512
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4,
        num_workers=0
    )

    #print(train_dataset[0])
    val_dataset = DATALoader(
        data=X_test,
        target=y_test,
        max_length=512
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=4,
        num_workers=0
    )

#     device = torch.device("cuda")
    device = torch.device(devName)
    model = BERTPrediction()

    if useSavedModel:
        print("Loading Saved Model...")
        checkpoint = torch.load(MODELPATH)
        model.load_state_dict(checkpoint['model_state_dict'])
    


    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias", 
        "LayerNorm,bias",
        "LayerNorm.weight",
               ]
    optimizer_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                   'weight_decay':0.001},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
                   'weight_decay':0.0}
    ]

    num_train_steps = int(len(X_train)/ 8*10)

    optimizers = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizers,
        num_warmup_steps=0,
        num_training_steps=num_train_steps

    )

    best_accuracy = -1
    for epoch in tqdm(range(epochs)):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizers, device=device, scheduler=scheduler, batchFraction=batchFraction)
        outputs, targets = eval_func(data_loader=train_data_loader, model=model, device=device, batchFraction=batchFraction)
        outputs = np.array(outputs) >= 0.5
        #print(f'\n\nTARGETS: {targets}\n\n')
        #print(f'OUTPUTS: {outputs}\n\n')
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score: {accuracy}")
        #import uuid
        #for when i create automate loading of the most recent checkpoint
        #{str(uuid.uuid4())[:8]}
        if accuracy>best_accuracy:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       MODELPATH)
            best_accuracy = accuracy
                


               
#run(devName="cuda", useSavedModel=True, epochs=1, batchFraction=0.01)#0 indicates only one batch
Inference(["Josh Jenkins is looking forward to TAB Breeders Crown Super Sunday https://t.co/antImqAo4Y https://t.co/ejnA78Sks0"])




