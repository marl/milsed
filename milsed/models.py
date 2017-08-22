# CREATED: 7/21/17 21:20 by Justin Salamon <justin.salamon@nyu.edu>

'''
MILSED models.

All model construction functions must implement the same API:

Parameters
----------
pump : Pump
    The Pump used to generate the features/labels
alpha : float
    The alpha parameter for the softmaxpool layer

Returns
-------
model
    Keras model
model_inputs
    Name(s) of input layer(s)
model_outputs
    Name(s) of output(s)

'''

import keras as K
import milsed.layers

from keras.layers import Dense, GRU,Bidirectional, Lambda, Conv1D
from keras.models import Model
# from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Multiply, Concatenate
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import TimeDistributed


def construct_crnn1d_smp(pump, alpha):
    '''
    CRNN with 1D conv encoder and Bi-GRU.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(conv1)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnn1d_max(pump, alpha):
    '''
    Max pooling.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(conv1)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = K.layers.GlobalMaxPooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnn1d_avg(pump, alpha):
    '''
    Mean pooling.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(conv1)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = K.layers.GlobalAveragePooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnn1d_smp(pump, alpha):
    '''
    CNN with 1D conv encoder and softmax pooling.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(conv1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnn1d_max(pump, alpha):
    '''
    CNN with 1D conv encoder and max pooling.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')
    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(conv1)

    p_static = K.layers.GlobalMaxPooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnn1d_avg(pump, alpha):
    '''
    CNN with 1D conv encoder and mean pooling.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')
    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(conv1)

    p_static = K.layers.GlobalAveragePooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnn2d_smp(pump, alpha):
    '''
    CRNN with 2D conv encoder and Bi-GRU.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: 3x3
    conv1 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)

    bn2 = K.layers.BatchNormalization()(conv1)

    conv2 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)

    bn3 = K.layers.BatchNormalization()(conv2)

    conv_sq = K.layers.Convolution2D(256, (1, 128),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn3)

    sq2 = milsed.layers.SqueezeLayer(axis=-2)(conv_sq)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnn1d2_smp(pump, alpha):
    '''
    CNN with 1D conv encoder and softmax pooling.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: a single 3-frame filters
    conv1 = K.layers.Convolution1D(64, 3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_sq)
                                   # data_format='channels_last')(x_sq)

    bn2 = K.layers.BatchNormalization()(conv1)

    conv2 = K.layers.Convolution1D(256, 9,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(conv2)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cbhg_smp(pump, alpha):
    '''CBHG Module
        input_height: non-time dim of input
        gru_d: size of gru hidden unit
        out_d: size of output
        out_actication: activation func for output e.g. 'Relu' (str)
        '''

    subtract_ones = Lambda(lambda x: K.backend.ones_like(x) - x)

    T = None
    K_bank = 16
    c = 128
    gru_d = 128
    out_activation = 'sigmoid'

    # Build the input layer
    # x = Input(shape=(T, input_height), dtype='float32')
    model_inputs = ['mel/mag']
    layers = pump.layers()
    x_input = layers['mel/mag']
    input_height = x_input.shape[-2].value

    x = milsed.layers.SqueezeLayer()(x_input)

    # Conv1D Bank
    Conv1D_filt_act = []
    for k in range(K_bank):
        Conv1D_filt_act.append(K.layers.Conv1D(c, k + 1, padding='same')(x))

    # Stack feature maps
    y = Concatenate()(Conv1D_filt_act)
    y = BatchNormalization()(y)

    # Max pooling
    y = MaxPooling1D(pool_size=2, strides=1, padding='same')(y)

    # Conv1D Projection
    y = Conv1D(input_height, 3, padding='same', activation='relu')(y)
    y = Conv1D(input_height, 3, padding='same', activation='linear')(y)
    y = BatchNormalization()(y)

    y = Add()([y, x])

    # Highway
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y)
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y_h)
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y_h)
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y_h)

    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(y)
    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(t_h)
    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(t_h)
    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(t_h)

    y_h = Multiply()([y_h, t_h])
    x_h = Multiply()([y, subtract_ones(t_h)])
    y = Add()([y_h, x_h])

    # Bi-directional GPU
    n_classes = pump.fields['static/tags'].shape[0]
    y = Bidirectional(GRU(gru_d, return_sequences=True), merge_mode='concat')(y)
    p_dynamic = TimeDistributed(Dense(n_classes, activation=out_activation),
                                name='dynamic/tags')(y)

    # Weak labels with SMP
    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model_outputs = ['dynamic/tags', 'static/tags']
    model = Model(inputs=[x_input], outputs=[p_dynamic, p_static])

    return model, model_inputs, model_outputs


def construct_crnn2d2_smp(pump, alpha):
    '''
    CRNN with 2D conv encoder and Bi-GRU with POOLING!

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: 3x3
    conv1 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)

    bn2 = K.layers.BatchNormalization()(conv1)
    p2 = K.layers.MaxPooling2D((2,2), strides=None, padding='valid')(bn2)

    conv2 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(p2)

    bn3 = K.layers.BatchNormalization()(conv2)
    p3 = K.layers.MaxPooling2D((2,2), strides=None, padding='valid')(bn3)

    conv_sq = K.layers.Convolution2D(256, (1, 32),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(p3)

    sq2 = milsed.layers.SqueezeLayer(axis=-2)(conv_sq)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnn2d3_smp(pump, alpha):
    '''
    CRNN with 2D conv encoder and Bi-GRU, smaller than crnn2d_smp

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: 3x3
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)

    bn2 = K.layers.BatchNormalization()(conv1)

    conv2 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)

    bn3 = K.layers.BatchNormalization()(conv2)

    conv_sq = K.layers.Convolution2D(64, (1, 128),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn3)

    sq2 = milsed.layers.SqueezeLayer(axis=-2)(conv_sq)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnn2d4_smp(pump, alpha):
    '''
    Like crnn2d3 but with batchnorm before the GRU

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # First convolutional filter: 3x3
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)

    bn2 = K.layers.BatchNormalization()(conv1)

    conv2 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)

    bn3 = K.layers.BatchNormalization()(conv2)

    conv_sq = K.layers.Convolution2D(64, (1, 128),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn3)

    bn4 = K.layers.BatchNormalization()(conv_sq)

    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn4)

    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_smp(pump, alpha):
    '''
    CRNN with L3-style conv encoder

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 1
    conv7 = K.layers.Convolution2D(512, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(512, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(1024, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_2_smp(pump, alpha):
    '''
    CRNN with L3-style conv encode, but smaller

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 1
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid')

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_3_smp(pump, alpha):
    '''
    Like crnnL3_2 but with L2 reg on bias term of output layer

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 1
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_4_smp(pump, alpha):
    '''
    Like crnnL3_3 but add another conv block (make deeper)

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # BLOCK 5
    conv9 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool8)
    bn9 = K.layers.BatchNormalization()(conv9)
    conv10 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn9)
    bn10 = K.layers.BatchNormalization()(conv10)
    pool10 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn10)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(512, (1, 4),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool10)
    bn11 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn11)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn1)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_5_smp(pump, alpha):
    '''
    Like crnnL3_3 but with 2 BiGRU layers

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn2)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_6_smp(pump, alpha):
    '''
    Like crnnL3_4 but has 2 BiGRU layers.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # BLOCK 5
    conv9 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool8)
    bn9 = K.layers.BatchNormalization()(conv9)
    conv10 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn9)
    bn10 = K.layers.BatchNormalization()(conv10)
    pool10 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn10)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(512, (1, 4),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool10)
    bn11 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn11)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn2)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_smp(pump, alpha):
    '''
    Like crnnL3_5 but with 3 BiGRU layers

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_max(pump, alpha):
    '''
    Like crnnL3_7_smp but max pooling at the end

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = K.layers.GlobalMaxPooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_avg(pump, alpha):
    '''
    Like crnnL3_7_smp but max pooling at the end

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = K.layers.GlobalAveragePooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_auto(pump, alpha):
    '''
    Like crnnL3_7 but with autopool instead of SMP

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.AutoPool(
        axis=1,
        kernel_constraint=K.constraints.non_neg(),
        kernel_regularizer=K.regularizers.l2(l=1e-4),
        name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_auto_2(pump, alpha):
    '''
    Like crnnL3_7_auto but with log/exp before/after autopool

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    log_layer = Lambda(lambda x: K.backend.log(x))
    exp_layer = Lambda(lambda x: K.backend.exp(x))

    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    # LOG input data to autopool
    p_log = log_layer(p_dynamic)

    p_static = milsed.layers.AutoPool(
        axis=1,
        kernel_constraint=K.constraints.non_neg(),
        kernel_regularizer=K.regularizers.l2(l=1e-4))(p_log)

    # EXP autopool output
    p_static_exp = milsed.layers.ExpLayer(name='static/tags')(p_static)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static_exp])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_auto_3(pump, alpha):
    '''
    Like crnnL3_7_autopool but no regularization on autopool layer

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.AutoPool(
        axis=1,
        kernel_constraint=K.constraints.non_neg(),
        name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_auto_4(pump, alpha):
    '''
    Like crnnL3_7_auto but without non-neg constraint

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.AutoPool(
        axis=1,
        kernel_regularizer=K.regularizers.l2(l=1e-4),
        name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_auto_5(pump, alpha):
    '''
    Like crnnL3_7_auto but with no constraint and no reg

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.AutoPool(
        axis=1,
        name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_8_smp(pump, alpha):
    '''
    Like crnnL3_6 but has 3 BiGRU layers.

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # BLOCK 5
    conv9 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool8)
    bn9 = K.layers.BatchNormalization()(conv9)
    conv10 = K.layers.Convolution2D(256, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn9)
    bn10 = K.layers.BatchNormalization()(conv10)
    pool10 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn10)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(512, (1, 4),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool10)
    bn11 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn11)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_9_smp(pump, alpha):
    '''
    Like crnnL3_7 but each block has 3 convs instead of 2

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    conv3 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)
    bn3 = K.layers.BatchNormalization()(conv3)
    pool3 = K.layers.MaxPooling2D((2,2), padding='valid')(bn3)

    # BLOCK 2
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool3)
    bn4 = K.layers.BatchNormalization()(conv4)
    conv5 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 3
    conv7 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    conv9 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn8)
    bn9 = K.layers.BatchNormalization()(conv9)
    pool9 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn9)

    # BLOCK 4
    conv10 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(pool9)
    bn10 = K.layers.BatchNormalization()(conv10)
    conv11 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn10)
    bn11 = K.layers.BatchNormalization()(conv11)
    conv12 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(bn11)
    bn12 = K.layers.BatchNormalization()(conv12)
    pool12 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn12)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool12)
    bnsq = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bnsq)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_10_smp(pump, alpha):
    '''
    Like crnnL3_8 but each block has 3 convs instead of 2

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    conv3 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)
    bn3 = K.layers.BatchNormalization()(conv3)
    pool3 = K.layers.MaxPooling2D((2,2), padding='valid')(bn3)

    # BLOCK 2
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool3)
    bn4 = K.layers.BatchNormalization()(conv4)
    conv5 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 3
    conv7 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    conv9 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn8)
    bn9 = K.layers.BatchNormalization()(conv9)
    pool9 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn9)

    # BLOCK 4
    conv10 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(pool9)
    bn10 = K.layers.BatchNormalization()(conv10)
    conv11 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn10)
    bn11 = K.layers.BatchNormalization()(conv11)
    conv12 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(bn11)
    bn12 = K.layers.BatchNormalization()(conv12)
    pool12 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn12)

    # BLOCK 5
    conv13 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(pool12)
    bn13 = K.layers.BatchNormalization()(conv13)
    conv14 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(bn13)
    bn14 = K.layers.BatchNormalization()(conv14)
    conv15 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(bn14)
    bn15 = K.layers.BatchNormalization()(conv15)
    pool15 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn15)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 4),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool15)
    bnsq = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bnsq)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_11_smp(pump, alpha):
    '''
    Like crnnL3_9 but each block has 5 convs instead of 3

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    conv3 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv3_2 = K.layers.Convolution2D(16, (3, 3),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn3)
    bn3_2 = K.layers.BatchNormalization()(conv3_2)
    conv3_3 = K.layers.Convolution2D(16, (3, 3),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn3_2)
    bn3_3 = K.layers.BatchNormalization()(conv3_3)
    pool3 = K.layers.MaxPooling2D((2,2), padding='valid')(bn3_3)

    # BLOCK 2
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool3)
    bn4 = K.layers.BatchNormalization()(conv4)
    conv5 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    conv6_2 = K.layers.Convolution2D(32, (3, 3),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn6)
    bn6_2 = K.layers.BatchNormalization()(conv6_2)
    conv6_3 = K.layers.Convolution2D(32, (3, 3),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn6_2)
    bn6_3 = K.layers.BatchNormalization()(conv6_3)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6_3)

    # BLOCK 3
    conv7 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    conv9 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn8)
    bn9 = K.layers.BatchNormalization()(conv9)
    conv9_2 = K.layers.Convolution2D(64, (3, 3),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn9)
    bn9_2 = K.layers.BatchNormalization()(conv9_2)
    conv9_3 = K.layers.Convolution2D(64, (3, 3),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer='he_normal')(bn9_2)
    bn9_3 = K.layers.BatchNormalization()(conv9_3)
    pool9 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn9_3)

    # BLOCK 4
    conv10 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(pool9)
    bn10 = K.layers.BatchNormalization()(conv10)
    conv11 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn10)
    bn11 = K.layers.BatchNormalization()(conv11)
    conv12 = K.layers.Convolution2D(128, (3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer='he_normal')(bn11)
    bn12 = K.layers.BatchNormalization()(conv12)
    conv12_2 = K.layers.Convolution2D(128, (3, 3),
                                      padding='same',
                                      activation='relu',
                                      kernel_initializer='he_normal')(bn12)
    bn12_2 = K.layers.BatchNormalization()(conv12_2)
    conv12_3 = K.layers.Convolution2D(128, (3, 3),
                                      padding='same',
                                      activation='relu',
                                      kernel_initializer='he_normal')(bn12_2)
    bn12_3 = K.layers.BatchNormalization()(conv12_3)
    pool12 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn12_3)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool12)
    bnsq = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bnsq)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_12_smp(pump, alpha):
    '''
    Like crnnL3_7 but with CBHG between last conv and GRU

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    ### INSERT CBHG HERE ###
    '''
    CBHG Module
    input_height: non-time dim of input
    gru_d: size of gru hidden unit
    out_d: size of output
    out_actication: activation func for output e.g. 'Relu' (str)
    '''

    subtract_ones = Lambda(lambda x: K.backend.ones_like(x) - x)

    T = None
    K_bank = 16
    c = 128
    gru_d = 128
    out_activation = 'sigmoid'

    # Build the input layer
    input_height = 256

    # x = milsed.layers.SqueezeLayer()(rnn3)

    # Conv1D Bank
    Conv1D_filt_act = []
    for k in range(K_bank):
        Conv1D_filt_act.append(K.layers.Conv1D(c, k + 1, padding='same')(rnn3))

    # Stack feature maps
    y = Concatenate()(Conv1D_filt_act)
    y = BatchNormalization()(y)

    # Max pooling
    y = MaxPooling1D(pool_size=2, strides=1, padding='same')(y)

    # Conv1D Projection
    y = Conv1D(input_height, 3, padding='same', activation='relu')(y)
    y = Conv1D(input_height, 3, padding='same', activation='linear')(y)
    y = BatchNormalization()(y)

    y = Add()([y, rnn3])

    # Highway
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y)
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y_h)
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y_h)
    y_h = TimeDistributed(Dense(input_height, activation='relu'))(y_h)

    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(y)
    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(t_h)
    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(t_h)
    t_h = TimeDistributed(Dense(input_height, activation='sigmoid'))(t_h)

    y_h = Multiply()([y_h, t_h])
    x_h = Multiply()([y, subtract_ones(t_h)])
    y = Add()([y_h, x_h])
    ### END CBHG HERE ###

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(y)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_crnnL3_7_smp_log(pump, alpha):
    '''
    Like crnnL3_7_smp but with log/exp before/after smp

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    log_layer = Lambda(lambda x: K.backend.log(x))
    exp_layer = Lambda(lambda x: K.backend.exp(x))

    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    # RNN
    # First recurrent layer: a 128-dim bidirectional gru
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(sq2)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn1)

    rnn3 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn2)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(rnn3)

    # LOG input data to autopool
    p_log = log_layer(p_dynamic)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1)(p_log)

    # EXP autopool output
    p_static_exp = milsed.layers.ExpLayer(name='static/tags')(p_static)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static_exp])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


# -- Pure convolution models
def construct_cnnL3_7_smp(pump, alpha):
    '''
    Like crnnL3_5 but without gru layers

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2)

    p_static = milsed.layers.SoftMaxPool(alpha=alpha,
                                         axis=1,
                                         name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnnL3_7_auto(pump, alpha):
    '''
    Like cnnL3_7 but with autopooling

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2)

    p_static = milsed.layers.AutoPool(axis=1,
                                      name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnnL3_7_avg(pump, alpha):
    '''
    Like cnnL3_7 but with avg pooling

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2)

    p_static = K.layers.GlobalAveragePooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


def construct_cnnL3_7_max(pump, alpha):
    '''
    Like cnnL3_7 but with max pooling

    Parameters
    ----------
    pump
    alpha

    Returns
    -------

    '''
    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # x_sq = milsed.layers.SqueezeLayer()(x_bn)

    # BLOCK 1
    conv1 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x_bn)
    bn1 = K.layers.BatchNormalization()(conv1)
    conv2 = K.layers.Convolution2D(16, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn1)
    bn2 = K.layers.BatchNormalization()(conv2)
    pool2 = K.layers.MaxPooling2D((2,2), padding='valid')(bn2)

    # BLOCK 2
    conv3 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool2)
    bn3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Convolution2D(32, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn3)
    bn4 = K.layers.BatchNormalization()(conv4)
    pool4 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn4)

    # BLOCK 3
    conv5 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool4)
    bn5 = K.layers.BatchNormalization()(conv5)
    conv6 = K.layers.Convolution2D(64, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn5)
    bn6 = K.layers.BatchNormalization()(conv6)
    pool6 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn6)

    # BLOCK 4
    conv7 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool6)
    bn7 = K.layers.BatchNormalization()(conv7)
    conv8 = K.layers.Convolution2D(128, (3, 3),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(bn7)
    bn8 = K.layers.BatchNormalization()(conv8)
    pool8 = K.layers.MaxPooling2D((2, 2), padding='valid')(bn8)

    # CONV SQUEEZE
    conv_sq = K.layers.Convolution2D(256, (1, 8),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='he_normal')(pool8)
    bn8 = K.layers.BatchNormalization()(conv_sq)
    sq2 = milsed.layers.SqueezeLayer(axis=-2)(bn8)

    n_classes = pump.fields['static/tags'].shape[0]

    p0 = K.layers.Dense(n_classes, activation='sigmoid',
                        bias_regularizer=K.regularizers.l2())

    p_dynamic = K.layers.TimeDistributed(p0, name='dynamic/tags')(sq2)

    p_static = K.layers.GlobalMaxPooling1D(name='static/tags')(p_dynamic)

    model = K.models.Model([x_mag],
                           [p_dynamic, p_static])

    model_outputs = ['dynamic/tags', 'static/tags']

    return model, model_inputs, model_outputs


MODELS = {'crnn1d_smp': construct_crnn1d_smp,
          'crnn1d_max': construct_crnn1d_max,
          'crnn1d_avg': construct_crnn1d_avg,
          'cnn1d_smp': construct_cnn1d_smp,
          'cnn1d_max': construct_cnn1d_max,
          'cnn1d_avg': construct_cnn1d_avg,
          'crnn2d_smp': construct_crnn2d_smp,
          'cnn1d2_smp': construct_cnn1d2_smp,
          'cbhg_smp': construct_cbhg_smp,
          'crnn2d2_smp': construct_crnn2d2_smp,
          'crnn2d3_smp': construct_crnn2d3_smp,
          'crnn2d4_smp': construct_crnn2d4_smp,
          'crnnL3_smp': construct_crnnL3_smp,
          'crnnL3_2_smp': construct_crnnL3_2_smp,
          'crnnL3_3_smp': construct_crnnL3_3_smp,
          'crnnL3_4_smp': construct_crnnL3_4_smp,
          'crnnL3_5_smp': construct_crnnL3_5_smp,
          'crnnL3_6_smp': construct_crnnL3_6_smp,
          'crnnL3_7_smp': construct_crnnL3_7_smp,
          'crnnL3_7_max': construct_crnnL3_7_max,
          'crnnL3_7_avg': construct_crnnL3_7_avg,
          'crnnL3_8_smp': construct_crnnL3_8_smp,
          'crnnL3_9_smp': construct_crnnL3_9_smp,
          'crnnL3_10_smp': construct_crnnL3_10_smp,
          'crnnL3_11_smp': construct_crnnL3_11_smp,
          'crnnL3_12_smp': construct_crnnL3_12_smp,
          'crnnL3_7_auto': construct_crnnL3_7_auto,
          'crnnL3_7_auto_2': construct_crnnL3_7_auto_2,
          'crnnL3_7_auto_3': construct_crnnL3_7_auto_3,
          'crnnL3_7_auto_4': construct_crnnL3_7_auto_4,
          'crnnL3_7_auto_5': construct_crnnL3_7_auto_5,
          'crnnL3_7_smp_log': construct_crnnL3_7_smp_log,
          'cnnL3_7_smp': construct_cnnL3_7_smp,
          'cnnL3_7_auto': construct_cnnL3_7_auto,
          'cnnL3_7_avg': construct_cnnL3_7_avg,
          'cnnL3_7_max': construct_cnnL3_7_max,
          }
