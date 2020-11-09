
from PIL import Image, ImageOps

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input

import pickle
import string
import streamlit as st
import tensorflow as tf

st.title("Image Caption")
st.text("Tell Caption Tell Caption Tell Caption Tell Caption")

new_model=tf.keras.models.load_model("my_model.h5")

def load_model():
  
  w2i_file=open("wordtoix.p","rb")
  wordtoix = pickle.load(w2i_file)

  i2w_file=open("ixtoword.p","rb")
  ixtoword = pickle.load(i2w_file)

  base_model = InceptionV3(weights = 'imagenet')

  model = Model(base_model.input, base_model.layers[-2].output)

with st.spinner("Loading Model Into Memory"):
  load_model()

max_length = 34

def preprocess_img(img):
  #inception v3 excepts img in 299*299

  #img = load_img(requests.get(img_path).content, target_size = (299, 299))
  x = img_to_array(img)
  # Add one more dimension
  x = np.expand_dims(x, axis = 0)
  x = preprocess_input(x)
  return x


def encode(im):
  image = preprocess_img(im)
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

file = st.file_uploader("Upload Photo",type=["jpg","png"])

if file is None:
  st.text("upoad image")
else:
  image = Image.open(file)
  st.image(image , use_colum_width = True)
  image = ImageOps.fit(image,(299,299),Image.ANTIALIAS)
  pic = encode(image)
  img = pic.reshape(1, 2048)
  final_caption = (greedy_search(img))
  st.write(final_caption)
  
if __name__ == "__main__":
  main()
