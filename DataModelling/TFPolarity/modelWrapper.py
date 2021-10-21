import re
import os
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
import nltk
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

class modelWrapper:
    def __init__(self, name):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = tensorflow.keras.models.load_model(name)
        self.labelEncoder = pickle.load(open(os.getcwd()+"\labelEncodingArray.pkl", 'rb'))
        self.wordVectorizer = pickle.load(open(os.getcwd()+"\wordEncodingTFIDF.pkl", 'rb'))
        self.textLabels = {"0":"negative", "4": "positive"}
    def inference(self, inputText):
        preprocessed = [preprocess(inputText)]
        input_vectorized = self.wordVectorizer.transform(preprocessed).toarray()
        prediction= self.model.predict(input_vectorized)
        label = decodeLabel(prediction[0][0], self.labelEncoder)
        sentence = decodeVector(input_vectorized, self.wordVectorizer)
        return {"label":self.textLabels[f'{label}'], "sentence":sentence, "raw_output":prediction[0]}




def decodeLabel(prediction, labelEncoding):
    prediction = int(round(prediction))
    label = labelEncoding[prediction]
    return label
def decodeVector(vector, vectorizer):
    output = vectorizer.inverse_transform(vector)
    return output
def preprocess(text):
    review = re.sub('[^a-zA-Z]',' ',text) 
    review = review.lower()
    review = review.split()
    ps = LancasterStemmer()
    try:
        all_stopwords = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        all_stopwords = stopwords.words('english')
        
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return ' '.join(review)