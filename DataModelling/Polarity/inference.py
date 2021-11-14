import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.utils import shuffle

import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn as nn
import warnings
# import xgboost
# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow import keras  `
# import tensorflow_text as text  # Registers the ops.

# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.models import Model
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense, Dropout, Input, Embedding


import os
from contextlib import contextmanager
import sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
            
            
HASHTAG_CHARS = r"\p{L}\p{M}\p{Nd}_\u200c\u200d\ua67e\u05be\u05f3\u05f4\uff5e\u301c\u309b\u309c\u30a0\u30fb\u3003\u0f0b\u0f0c\u00b7"
MODELPATH = os.getcwd() + "\\checkpoints\\model.bin"

import html
import regex
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


def Inference(rawTweetsList, devName="cpu"):
    with suppress_stdout():
        processedTweetList = [ process_tweet(tweet) for tweet in rawTweetsList]
        
        
        device = torch.device(devName)
        model = BERTPrediction()
        checkpoint = torch.load(MODELPATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        tempDataset = DATALoader(data=processedTweetList, target=[], max_length=512)
        #print(f"\n\n\n\nLength: {len(rawTweetsList)}")
        infLoader = torch.utils.data.DataLoader(tempDataset, batch_size=len(rawTweetsList), num_workers=0)
        outputs = []
        with torch.no_grad():
           for bi, d in enumerate(infLoader):
               ids = d["ids"]
               token_type_ids = d["token_type_ids"]
               mask = d["mask"]
    
               ids = ids.to(device, dtype=torch.long)
               token_type_ids = token_type_ids.to(device, dtype=torch.long)
               mask = mask.to(device, dtype=torch.long)
    
    
               output = model(
                   ids=ids,
                   masks = mask,
                   token_type_ids = token_type_ids
               )
               output = (torch.sigmoid(output).cpu().detach().numpy().tolist())[0]
    
               output = [round(out,4) for out in output]
    
               outputs.append(output)
    
    infPrint(processedTweetList, outputs)
    return outputs
    
def infPrint(tweets, scores):
    print("\n\n")
    outList = zip(tweets, scores)
    for output in outList:
        print(f'TWEET: \"{output[0]}\"')
        print(f'{round(100*output[1][0], 4)}% Negative, {round(100 * output[1][1], 4)}% Neutral, {round(100 * output[1][2], 4)}% Positive,')
        

               





