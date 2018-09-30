import numpy as np
import pandas as pd

import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.layers as layers

from keras.models import Sequential, Model
from keras.layers import Dense,Activation, Dropout, Input
from keras.layers import Conv2D,MaxPool2D
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# import data
from prepare_data import load_train_data, load_val_data, load_img, load_img2, load_word, tfidf

# import model
from model import build_vgg16, initialize, attend, decode, rnn_and_loss

#Parameters for RNN
vocab_size = 5000
dim_embedding = 512
max_caption_length = 20

num_lstm_units = 512
vocabulary_size = 5000

num_ctx = 196 # 有196個context vector，每一張圖萃取出196個region，每一個region用一個vector表示
dim_ctx = 512 

fc_drop_rate = 0.5
lstm_drop_rate = 0.3
attention_loss_factor = 0.01

fc_kernel_initializer_scale = 0.08
fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -fc_kernel_initializer_scale,
            maxval = fc_kernel_initializer_scale)

is_train = True

fc_kernel_regularizer_scale = 1e-4
if fc_kernel_regularizer_scale > 0:
    fc_kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = fc_kernel_regularizer_scale)
else:
    fc_kernel_regularizer = None
    
# 只選取一部分的 data 來試著訓練 (因為還沒買 GPU...)
df = pd.read_csv('annotations.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df[:100]
img_ids = np.array(df['image_id'])
image_files = np.array(df['image_file'])
captions = np.array(df['caption'])

cap_padded, masks = load_word(captions)

batch_size = 32
image_shape = [224,224,3]

kernel_size = (3,3)
strides = (1,1)

optimizer = tf.train.AdamOptimizer()

dataset = tf.data.Dataset.from_tensor_slices((image_files, cap_padded, masks, img_ids)).map(load_img2)
dataset = dataset.batch(32, drop_remainder=True)       

contexts = []

for (batch, (images, path, cap_padded, masks, img_ids)) in enumerate(dataset):
    if images.shape[0] == 32:
        context = build_vgg16(images, batch_size, image_shape, kernel_size, strides).numpy()
        contexts.append(context)
    else:
        cnt = 32-images.shape[0]
        
        image_files_ = image_files[3:cnt+3]
        cap_padded = cap_padded[3:cnt+3]
        masks = masks[3:cnt+3]
        img_ids = img_ids[3:cnt+3]
    
        images_ =images
        
        for img in image_files_:
            img, path, cap, mask, img_id = load_img2(img, cap_padded, masks, img_ids)
            img = tf.reshape(img,(1, 224, 224, 3))         
            images_ = tf.concat([images_, img], axis = 0)
        context_ = build_vgg16(images_, batch_size, image_shape, kernel_size, strides).numpy()
        contexts.append(context_)
        
    all_preds = []
    all_total_loss= []
    all_accu = []
    
    for i in range(len(np.asarray(contexts))):
        
        predictions, cross_entropies, alphas, predictions_correct = rnn_and_loss(contexts[i], cap_padded, masks)

        cross_entropies = tf.stack(cross_entropies, axis=1)
        cross_entropy_loss = tf.reduce_sum(cross_entropies)/tf.reduce_sum(masks)

        alphas = tf.stack(alphas, axis=1)
        alphas = tf.reshape(alphas, [batch_size, num_ctx, -1])
        attentions = tf.reduce_sum(alphas, axis=2)
        diffs = tf.ones_like(attentions) - attentions
        attention_loss = attention_loss_factor* tf.nn.l2_loss(diffs)/(batch_size * num_ctx)

        reg_loss = tf.losses.get_regularization_loss()

        total_loss = cross_entropy_loss #+ attention_loss + reg_loss

        predictions_correct = tf.stack(predictions_correct, axis=1)
        accuracy = tf.reduce_sum(predictions_correct)/ tf.reduce_sum(masks)
        optimizer.minimize(total_loss)
        
        all_preds.append(predictions)
        all_total_loss.append(total_loss)
        all_accu.append(accuracy)
    
        
        
