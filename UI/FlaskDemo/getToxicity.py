# USING TENSORFLOW 2.4.1 uinsg --user tag

from transformers import TFBertModel
import os
import numpy as np
import bert_utils as bert_utils
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

input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
input_type = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
input_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
inputs = [input_ids, input_type, input_mask]
bert = TFBertModel.from_pretrained('bert-base-uncased')
bert_outputs = bert(inputs)
print(bert_outputs)
last_hidden_states = bert_outputs[0]
avg = layers.GlobalAveragePooling1D()(last_hidden_states)
output = layers.Dense(N_CLASSES, activation="sigmoid")(avg)
model = keras.Model(inputs=inputs, outputs=output)
print(f'CurrDir: {os.getcwd()}')
model.load_weights(os.getcwd() + '\\toxicityModel\\toxmodelckpt')


def getToxicity(sentences):
    enc_sentences = bert_utils.prepare_bert_input(sentences, MAX_SEQ_LEN, 'bert-base-uncased')
    predictions = model.predict(enc_sentences)

    sentence_tox_pairs = []
    for sentence, pred in zip(sentences, predictions):
        mask = (pred > 0.5).astype(bool)
        tox = str(classes[mask]) if any(mask) else "not toxic"
        sentence_tox_pairs.append((sentence.replace("\n", "").strip(), tox, pred[mask]))
    return sentence_tox_pairs