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

annotation_file = 'captions_train2014.json'
PATH = './train2014/'

def load_data():
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

        all_captions=[]
        img_name_vector =[]
        image_ids = []
        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)
            image_ids.append(image_id)

    train_captions, img_name_vector, image_ids = shuffle(all_captions,
                                              img_name_vector,
                                              image_ids, 
                                              random_state=1)
    return train_captions, img_name_vector, image_ids

# train_captions, img_name_vector, image_ids = load_data()

# resize pictures by tensorflow, ouput would be 'tf.tensor'
def load_img(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (224, 224))
    img = tf.image.per_image_standardization(img)
    return img, image_path

def load_word(train_captions):
    top_k = 5000

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                      oov_token="<unk>", 
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post', maxlen=20)


    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value<5000}
    tokenizer.word_index[tokenizer.oov_token] = top_k + 1
    tokenizer.word_index['<pad>'] = 0

    index_word = {value:key for key, value in tokenizer.word_index.items()}

    tfid_vectorizer = TfidfVectorizer('english')
    tfid_vectorizer.fit(train_captions)
    dictionary = tfid_vectorizer.vocabulary_.items()
    vocabulary = {key:value for key, value in dictionary if value < 5000}
    
    # Prepare masks
    masks = []
    for caption in train_seqs:
        current_num_words = len(caption)
        current_masks = np.zeros(20) # max_caption_length = 20
        current_masks[:current_num_words] = 1.0
        masks.append(current_masks)
    masks = np.array(masks)
    
    return cap_padded, masks

#sentences, masks = load_word(train_captions)


