import tensorflow as tf

import json
import pickle
import os
import pandas as pd
import numpy as np

# Scikit-learn includes many helpful utilities
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_train_data():
    train_annotation_file = 'captions_train2014.json'
    PATH = './train2014/'
    with open(train_annotation_file, 'r') as f:
        annotations = json.load(f)

        train_captions=[]
        train_img_name =[]
        train_image_ids = []
        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            train_img_name.append(full_coco_image_path)
            train_captions.append(caption)
            train_image_ids.append(image_id)

    train_captions, train_img_name, train_image_ids = shuffle(train_captions,
                                              train_img_name,
                                              train_image_ids, 
                                              random_state=1)
    return train_captions, train_img_name, train_image_ids

def load_val_data():
    val_annotation_file = 'captions_val2014.json'
    PATH = './val2014/'
    with open(val_annotation_file, 'r') as f:
        annotations = json.load(f)

        val_captions=[]
        val_img_name =[]
        val_image_ids = []
        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)

            val_img_name.append(full_coco_image_path)
            val_captions.append(caption)
            val_image_ids.append(image_id)

    val_captions, val_img_name, val_image_ids = shuffle(val_captions,
                                              val_img_name,
                                              val_image_ids, 
                                              random_state=1)
    return val_captions, val_img_name, val_image_ids



def load_word(captions):
    top_k = 5000

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                      oov_token="<unk>", 
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    _seqs = tokenizer.texts_to_sequences(captions)
    cap_padded = tf.keras.preprocessing.sequence.pad_sequences(_seqs, padding='post', maxlen=20)


    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value<5000}
    tokenizer.word_index[tokenizer.oov_token] = top_k + 1
    tokenizer.word_index['<pad>'] = 0

    index_word = {value:key for key, value in tokenizer.word_index.items()}
    
    # Prepare masks
    masks = []
    for caption in _seqs:
        current_num_words = len(caption)
        current_masks = np.zeros(20) # max_caption_length = 20
        current_masks[:current_num_words] = 1.0
        masks.append(current_masks)
    masks = np.array(masks)
    masks = tf.cast(masks, tf.float32)
    
    return cap_padded, masks

#sentences, masks = load_word(train_captions)

def tfidf(captions):
    tfid_vectorizer = TfidfVectorizer('english')
    tfid_vectorizer.fit(captions)
    dictionary = tfid_vectorizer.vocabulary_.items()
    vocabulary = {key:value for key, value in dictionary if value < 5000}
    
    return vocabulary

# resize pictures by tensorflow, ouput would be 'tf.tensor'
def load_img2(image_path, cap_padded, masks, img_ids):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (224, 224))
    img = tf.image.per_image_standardization(img)
    return img, image_path, cap_padded, masks, img_ids

def load_img(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (224, 224))
    img = tf.image.per_image_standardization(img)
    return img
    
    
   


