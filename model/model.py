# import numpy as np
import warnings
from deeplift.layers import BatchNormalization
from keras.layers import SpatialDropout1D

warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras
from tensorflow.keras.layers import Conv1D
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import CuDNNLSTM,CuDNNGRU
import keras.backend as K
from keras import backend as K
MAX_LEN = 174
def load_csv(path):
    data_read = pd.read_csv(path,index_col=0,header=None)
    data = np.array(data_read)
    #print(data.shape)
    # print(data)
    return data
embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])


def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def ResBlock1(x,filters,kernel_size1,kernel_size2,dilation_rate):
    #r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(x)#第一卷积
    r1 = Conv1D(filters, kernel_size1, padding='same', dilation_rate=dilation_rate)(x)
    r2 = Conv1D(filters, kernel_size2, padding='same', dilation_rate=dilation_rate)(x)
    r1 = BatchNormalization()(r1)
    #r = SpatialDropout1D(0.2)(r)
    r1 = Dropout(0.2)(r1)
    r1 = Activation('relu')(r1)
    r2 = BatchNormalization()(r2)
    #r = SpatialDropout1D(0.2)(r)
    r2 = Dropout(0.2)(r2)
    r2 = Activation('relu')(r2)
    r = concatenate([r1, r2])
    r3 = Conv1D(filters, kernel_size1, padding='same', dilation_rate=dilation_rate)(r)
    r4 = Conv1D(filters, kernel_size2, padding='same', dilation_rate=dilation_rate)(r)
    r3= BatchNormalization()(r3)
    #r = SpatialDropout1D(0.2)(r)
    r3 = Dropout(0.2)(r3)
    r3 = Activation('relu')(r3)
    r4= BatchNormalization()(r4)
    r4 = Dropout(0.2)(r4)
    r4 = Activation('relu')(r4)
    r = concatenate([r3, r4],name='concat%d'%dilation_rate)
    #r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(r)#第二卷积
    #r = SpatialDropout1D(0.2)(r)
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters*2,1,padding='same')(x)  #shortcut（捷径）
    o=add([r,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o
def ResBlock(x,filters,kernel_size,dilation_rate):
    #r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(x)#第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    r = BatchNormalization()(r)
    #r = SpatialDropout1D(0.2)(r)
    r = Dropout(0.2)(r)
    r = Activation('relu')(r)
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)
    r = BatchNormalization()(r)
    #r = SpatialDropout1D(0.2)(r)
    r = Dropout(0.2)(r)
    r = Activation('relu')(r)
    #r = tfa.layers.WeightNormalization(Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu'))(r)#第二卷积
    #r = SpatialDropout1D(0.2)(r)
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,1,padding='same')(x)  #shortcut（捷径）
    o=add([r,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o

class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init =tf.compat.v1.keras.initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
def CNN_GRU_ATT_model(layers=2, filters=16,
                     growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = Input(shape=(MAX_LEN,))

    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                       trainable=False)(sequence)
    conv_layer1 = Convolution1D(
        filters=64,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer1 = MaxPooling1D(pool_size=int(2))
    conv_layer2 = Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer2 = MaxPooling1D(pool_size=int(2))
    conv_layer3 = Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer3 = MaxPooling1D(pool_size=int(2))


    enhancer_branch = Sequential()
    enhancer_branch.add(conv_layer1)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(max_pool_layer1)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    enhancer_branch.add(conv_layer2)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(max_pool_layer2)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    #enhancer_branch.add(conv_layer3)
    #enhancer_branch.add(Activation("relu"))
    #enhancer_branch.add(max_pool_layer3)
    #enhancer_branch.add(BatchNormalization())
    #enhancer_branch.add(Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)
    l_gru1 = Bidirectional(GRU(16, return_sequences=True))(enhancer_out)
    x = AttLayer(16)(l_gru1)

    dt = Dense(32)(x)  # kernel_initializer="glorot_uniform"
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([sequence], preds)
    return model
def deepires_model(layers=2, filters=16,
                     growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = Input(shape=(MAX_LEN,))
    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                       trainable=False)(sequence)
    x = ResBlock1(emb_en, filters=16, kernel_size1=2,kernel_size2=3, dilation_rate=1)
    x = ResBlock1(x, filters=8, kernel_size1=2,kernel_size2=3,dilation_rate=2)
    l_gru1 = Bidirectional(GRU(8, return_sequences=True))(x)
    x = AttLayer(8)(l_gru1)

    dt = Dense(32)(x)  # kernel_initializer="glorot_uniform"
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([sequence], preds)
    return model
def CNN_GRU_model(layers=2, filters=16,
                     growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = Input(shape=(MAX_LEN,))

    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                       trainable=False)(sequence)
    conv_layer1 = Convolution1D(
        filters=32,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer1 = MaxPooling1D(pool_size=int(2))
    conv_layer2 = Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer2 = MaxPooling1D(pool_size=int(2))
    conv_layer3 = Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer3 = MaxPooling1D(pool_size=int(2))


    enhancer_branch = Sequential()
    enhancer_branch.add(conv_layer1)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(max_pool_layer1)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    enhancer_branch.add(conv_layer2)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(max_pool_layer2)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    #enhancer_branch.add(conv_layer3)
    #enhancer_branch.add(Activation("relu"))
    #enhancer_branch.add(max_pool_layer3)
    #enhancer_branch.add(BatchNormalization())
    #enhancer_branch.add(Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)
    l_gru1 = Bidirectional(GRU(16, return_sequences=True))(enhancer_out)
    x = Flatten()(l_gru1)

    dt = Dense(32)(x)  # kernel_initializer="glorot_uniform"
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([sequence], preds)
    return model

def CNN_model(layers=2, filters=16,
                     growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = Input(shape=(MAX_LEN,))

    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                       trainable=False)(sequence)
    conv_layer1 = Convolution1D(
        filters=64,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer1 = MaxPooling1D(pool_size=int(2))
    conv_layer2 = Convolution1D(
        filters=32,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer2 = MaxPooling1D(pool_size=int(2))
    conv_layer3 = Convolution1D(
        filters=16,
        kernel_size=3,
        padding="same",  # "same"
    )
    max_pool_layer3 = MaxPooling1D(pool_size=int(2))


    enhancer_branch = Sequential()
    enhancer_branch.add(conv_layer1)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(max_pool_layer1)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    enhancer_branch.add(conv_layer2)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(max_pool_layer2)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    #enhancer_branch.add(conv_layer3)
    #enhancer_branch.add(Activation("relu"))
    #enhancer_branch.add(max_pool_layer3)
    #enhancer_branch.add(BatchNormalization())
    #enhancer_branch.add(Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)
    flatten= Flatten()(enhancer_out)
    dt1 = Dropout(0.2)(flatten)

    dt = Dense(64)(dt1)  # kernel_initializer="glorot_uniform"
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([sequence], preds)
    return model

def GRU_model(layers=2, filters=16,
                             growth_rate=8, dropout_rate=0.2, weight_decay=1e-4):
    sequence = Input(shape=(MAX_LEN,))
    # enhancers1 = Input(shape=(MAX_LEN_en,7))

    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                       trainable=False)(sequence)


    # x_multiply2 = SimAM()(enhancer_out)
    l_gru1 = Bidirectional(GRU(16, return_sequences=True))(emb_en)
    #l_gru2 = Bidirectional(GRU(8, return_sequences=True))(l_gru1)
    x = AttLayer(16)(l_gru1)

    bn2 = BatchNormalization()(x)

    dt1 = Dropout(0.2)(bn2)


    dt = Dense(64)(dt1)  # kernel_initializer="glorot_uniform"
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([sequence], preds)
    # adam = tensorflow.keras.optimizers.Adam(lr=2e-6)
    # sgd=tensorflow.keras.optimizers.SGD(lr=8e-6)

    return model




