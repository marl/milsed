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


MODELS = {'crnn1d_smp': construct_crnn1d_smp,
          'crnn1d_max': construct_crnn1d_max,
          'crnn1d_avg': construct_crnn1d_avg,
          'cnn1d_smp': construct_cnn1d_smp,
          'cnn1d_max': construct_cnn1d_max,
          'cnn1d_avg': construct_cnn1d_avg,
          'crnn2d_smp': construct_crnn2d_smp,
          'cnn1d2_smp': construct_cnn1d2_smp,
          'cbhg_smp': construct_cbhg_smp}

