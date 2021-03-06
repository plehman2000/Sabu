#USING TENSORFLOW 2.4.1 uinsg --user tag

from transformers import TFBertModel
import os
import numpy as np
#from DataModelling.getTox import bert_utils as bert_utils
np.set_printoptions(precision=2)
import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import ModelCheckpoint


N_CLASSES = 6
MAX_SEQ_LEN = 128
BERT_NAME = 'bert-base-uncased'
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
classes = np.array(classes)


from transformers import BertTokenizer


def prepare_bert_input(sentences, seq_len, bert_name):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    encodings = tokenizer(sentences, truncation=True, padding='max_length',
                                max_length=seq_len)
    input = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),
               np.array(encodings["attention_mask"])]
    return input


input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
input_type = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
input_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
inputs = [input_ids, input_type, input_mask]
bert = TFBertModel.from_pretrained('bert-base-uncased')
bert_outputs = bert(inputs)
last_hidden_states = bert_outputs.last_hidden_state
avg = layers.GlobalAveragePooling1D()(last_hidden_states)
output = layers.Dense(N_CLASSES, activation="sigmoid")(avg)
model = keras.Model(inputs=inputs, outputs=output)
print(f'CurrDir: {os.getcwd()}')
model.load_weights(os.getcwd() + '\\DataModelling\\getTox\\toxicityModel\\toxmodelckpt')

def toxInference(sentences):
    enc_sentences = prepare_bert_input(sentences, MAX_SEQ_LEN, 'bert-base-uncased')
    predictions = model.predict(enc_sentences)
    
    toxicityList = []
    for sentence, pred in zip(sentences, predictions):
        mask = (pred > 0.5).astype(bool)
        tox = str(classes[mask]) if any(mask) else "not toxic"
        toxicityList.append([tox, pred[mask]])
    return toxicityList