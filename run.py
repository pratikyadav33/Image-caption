import os 
import pickle
import tensorflow as tf
import numpy as np
    
import requests
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask, redirect, url_for, request, render_template
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app)

#Loadning Necessary Files And Documents
new_model= load_model("/my-cap.h5")
w2i_file = open("/wordtoix.p","rb")
wordtoix = pickle.load(w2i_file)

i2w_file=open("/ixtoword.p","rb")
ixtoword = pickle.load(i2w_file)

base_model = InceptionV3(weights = 'imagenet')

model = Model(base_model.input, base_model.layers[-2].output)

max_length = 34

print('Model loaded')

def preprocess_img(img_path):
    img = load_img(img_path, target_size = (299, 299))
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess_img(image)
    vec = model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec
    
def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = new_model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/',methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET",'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = '/'
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pic = encode(file_path)
        img = pic.reshape(1, 2048)
        caption = greedy_search(img)

        pic = encode(file_path)
        img = pic.reshape(1, 2048)
        caption = greedy_search(img)
        
        os.remove(file_path)
        return caption
    return None

if __name__ == "__main__":
    app.run()
