# CREATED: 6/12/17 13:45 by Justin Salamon <justin.salamon@nyu.edu>

from keras import backend as K
from keras.engine.topology import Layer


class SoftMaxPool(Layer):
    '''
    Keras softmax pooling layer
    '''

    def __init__(self, axis=-1, **kwargs):
        super(SoftMaxPool, self).__init__(**kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        shape[self.axis] = 1
        return tuple(shape)

    def call(self, x, mask=None):
        m = K.max(x, axis=self.axis, keepdims=True)
        sm = K.exp(x - m)
        w = sm / K.sum(sm, axis=self.axis, keepdims=True)
        return K.sum(x * w, axis=self.axis, keepdims=True)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SoftMaxPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SqueezeLayer(Layer):
    '''
    Keras squeeze layer
    '''
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        # shape = np.array(input_shape)
        # shape = shape[shape != 1]
        # return tuple(shape)
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def call(self, x, mask=None):
        return K.squeeze(x, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BagToBatchLayer(Layer):
    '''
    Convert bags with frame-level predictions to a batch where each sample in
    the batch corresponds to a single frame.
    '''
    def __init__(self, **kwargs):
        super(BagToBatchLayer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[1]
        if input_shape[0] is not None:
            shape[0] *= input_shape[1]
        return tuple(shape)

    def call(self, x, mask=None):
        return K.reshape(x, self.get_output_shape_for(x.shape))

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(BagToBatchLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
