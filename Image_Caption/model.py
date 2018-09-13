
# Parameters for CNN
batch_size = 32
image_shape = [224,224,3]

kernel_size = (3,3)
strides = (1,1)

def build_vgg16(images, batch_size, image_shape, kernel_size, strides):

    conv1_1_feats = tf.layers.conv2d(images, 64, kernel_size, strides, padding ='same', 
                                     activation =tf.nn.relu, use_bias = True, 
                                     name = 'conv1_1')
    conv1_2_feats = tf.layers.conv2d(conv1_1_feats, 64, kernel_size, strides, padding ='same', 
                                     activation =tf.nn.relu, use_bias = True, 
                                     name = 'conv1_2')
    pool1_feats = tf.layers.max_pooling2d(conv1_2_feats, pool_size=2, strides=2, name = 'pool1')

    conv2_1_feats = tf.layers.conv2d(pool1_feats, 128, kernel_size, strides, padding ='same', 
                                     activation =tf.nn.relu, use_bias = True, 
                                     name = 'conv2_1')
    conv2_2_feats = tf.layers.conv2d(conv2_1_feats, 128,kernel_size, strides, padding ='same', 
                                     activation =tf.nn.relu, use_bias = True, 
                                     name = 'conv2_2')
    pool2_feats = tf.layers.max_pooling2d(conv2_2_feats, pool_size=2, strides =2, name = 'pool2')

    conv3_1_feats = tf.layers.conv2d(pool2_feats, 256, kernel_size, strides, padding ='same', 
                                     activation =tf.nn.relu, use_bias = True, 
                                     name = 'conv3_1')
    conv3_2_feats = tf.layers.conv2d(conv3_1_feats, 256, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv3_2')
    conv3_3_feats = tf.layers.conv2d(conv3_2_feats, 256, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv3_3')
    pool3_feats = tf.layers.max_pooling2d(conv3_3_feats, pool_size=2, strides =2, name = 'pool3')

    conv4_1_feats = tf.layers.conv2d(pool3_feats, 512, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv4_1')
    conv4_2_feats = tf.layers.conv2d(conv4_1_feats, 512, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv4_2')
    conv4_3_feats = tf.layers.conv2d(conv4_2_feats, 512, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv4_3')
    pool4_feats = tf.layers.max_pooling2d(conv4_3_feats, pool_size=2, strides =2, name = 'pool4')

    conv5_1_feats = tf.layers.conv2d(pool4_feats, 512, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv5_1')
    conv5_2_feats = tf.layers.conv2d(conv5_1_feats, 512, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv5_2')
    conv5_3_feats = tf.layers.conv2d(conv5_2_feats, 512, kernel_size, strides, padding ='same', activation =tf.nn.relu, use_bias = True, name = 'conv5_3')
    
    reshaped_conv5_3_feats = tf.reshape(conv5_3_feats, 
                                        [batch_size, 196, 512])

    conv_feats = reshaped_conv5_3_feats
    return conv_feats 


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

"""以context_mean來初始化"""
def initialize(cont_mean):
    context_mean = tf.layers.dropout(inputs = cont_mean, rate = fc_drop_rate, training = is_train)
    ##fc_drop_rate = 0.5;is_train = True
    memory = tf.layers.dense(cont_mean, units=num_lstm_units,
                             activation = None,
                             use_bias = True,
                             trainable = is_train,
                             activity_regularizer = None)
                           
    output = tf.layers.dense(cont_mean,
                           units=num_lstm_units,
                           activation=None,
                           use_bias = True,
                           trainable = is_train,
                           activity_regularizer = None)
    return memory, output

# attention
"""
1. calculate match score
2. put match score into softmax layer to obtain alpha (seen as probability)

contexts = conv_feats
維度為 32 x 196 x 512 

output 是 last output
維度為 32 * 512
"""

def attend(contexts, output):
    reshaped_context = tf.reshape(contexts, [-1,dim_ctx]) # 6272 * 512
    reshaped_context = tf.layers.dropout(reshaped_context, 
                                      rate = fc_drop_rate)
    output = tf.layers.dropout(output, fc_drop_rate) # 32 * 512
  
    logits1 = tf.layers.dense(reshaped_context, 
                           units = 1,
                           activation = None,
                           use_bias = False)
                            # after shape 6272 * 1
    logits1 = tf.reshape(logits1, [-1, num_ctx]) #  32 * 196
  
    logits2 = tf.layers.dense(output, 
                           units = num_ctx, 
                           activation = None,
                           use_bias = False) #  32 * 196
    logits = logits1 + logits2 # 32 * 196
  
    alpha = tf.nn.softmax(logits) # 32 * 196
  
    return alpha # 32 * 196

def decode(expanded_output):
    """ Decode the expanded output of the LSTM into a word. """
    expanded_output = tf.layers.dropout(expanded_output)
 
    logits = tf.layers.dense(expanded_output,
                               units = vocabulary_size,
                               activation = None,
                               name = 'fc')
    return logits

def rnn_and_loss(contexts, senteces, masks):
    last_memory = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, num_lstm_units]) # 32 * 512

    last_output = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, num_lstm_units]) # 32 * 512

    last_word = tf.placeholder(
        dtype=tf.int32,
        shape=[batch_size]) # 32
    
    with tf.variable_scope('word_embedding'):
        embedding_matrix = tf.get_variable(shape=[vocab_size,dim_embedding],
                                    initializer=fc_kernel_initializer,                                    
                                    trainable=is_train,
                                    name = 'weights')
    # Initialize the LSTM using the mean context
    with tf.variable_scope("initialize"):
    #     context_mean = tf.reduce_mean(conv_feats, axis=1) # after shape 32 * 512
        initial_memory, initial_output = initialize(tf.reduce_mean(conv_feats, axis=1))
        initial_state = initial_memory, initial_output # 32 * 512
    lstm = tf.nn.rnn_cell.LSTMCell(
            num_lstm_units,
            initializer=fc_kernel_initializer)

    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,
      input_keep_prob=1.0 -lstm_drop_rate,
      output_keep_prob=1.0 - lstm_drop_rate,
      state_keep_prob=1.0 - lstm_drop_rate)
    
#     prepare to run
    predictions =[]
    cross_entropies = []

    alphas = []
    predictions_correct = []

    num_steps = max_caption_length

    last_output = initial_output
    last_memory = initial_memory
    last_word = tf.zeros([batch_size], tf.int32)

    last_state = last_memory, last_output
    
    for idx in range(num_steps):
        with tf.variable_scope('attend', reuse=tf.AUTO_REUSE):
            alpha = attend(contexts, last_output)
            """attention的第三個步驟： 
            contexts shape == 32 * 196 * 512; 
            alpha shape == 32 * 196
            alpha擴展第三個維度 ＝＝ (32*196*1)
            將alpha值乘上 contexts
            """
            context = tf.reduce_sum(contexts * tf.expand_dims(alpha,2), axis = 1) # after shape 32 * 512

            tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                  [1, 196])
            masked_alpha = alpha * tiled_masks
            alphas.append(tf.reshape(masked_alpha, [-1]))

      # Embed the last word
        with tf.variable_scope("word_embedding"):
            word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                              last_word)

      # Apply the LSTM
        with tf.variable_scope("lstm"):
            current_input = tf.concat([context, word_embed], 1)
            output, state = lstm(current_input, last_state)
            memory, _ = state

      # Decode the expanded output of LSTM into a word
        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
            expanded_output = tf.concat([output,
                                       context,
                                       word_embed],
                                      axis=1)
            logits = decode(expanded_output)
            probs = tf.nn.softmax(logits)

            prediction = tf.argmax(logits, 1)
            predictions.append(prediction)

            # Compute the loss for this step
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=sentences[:, idx],
              logits=logits)
            masked_cross_entropy = cross_entropy * masks[:, idx]
            cross_entropies.append(masked_cross_entropy)

            ground_truth = tf.cast(sentences[:, idx], tf.int64)
            prediction_correct = tf.where(
                  tf.equal(prediction, ground_truth),
                  tf.cast(masks[:, idx], tf.float32),
                  tf.cast(tf.zeros_like(prediction), tf.float32))
            predictions_correct.append(prediction_correct)

            last_output = output
            last_memory = memory
            last_state = state
            last_word = sentences[:, idx]

        tf.get_variable_scope().reuse_variables() 
        return predictions, cross_entropies # both are lists 
