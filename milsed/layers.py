# CREATED: 6/12/17 13:45 by Justin Salamon <justin.salamon@nyu.edu>

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import constraints
from keras import regularizers


class SoftMaxPool(Layer):
    '''
    Keras softmax pooling layer.
    '''

    def __init__(self, axis=0, alpha=1.0, **kwargs):
        '''

        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0
            (should be time).
        alpha : float
            alpha = 1: softmax pooling
            alpha = 0: mean pooling
            alpha = large: converges to max pooling
        kwargs
        '''
        super(SoftMaxPool, self).__init__(**kwargs)

        self.axis = axis
        self.alpha = alpha

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        m = K.max(x, axis=self.axis, keepdims=True)
        sm = K.exp(self.alpha * (x - m))
        w = sm / K.sum(sm, axis=self.axis, keepdims=True)
        return K.sum(x * w, axis=self.axis, keepdims=False)

    def get_config(self):
        config = {'axis': self.axis, 'alpha': self.alpha}
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

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

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


class AutoPool(Layer):
    '''Automatically tuned soft-max pooling.

    This layer automatically adapts the pooling behavior to interpolate
    between mean- and max-pooling for each dimension.
    '''
    def __init__(self, axis=0,
                 kernel_initializer='ones',
                 kernel_constraint=None,
                 kernel_regularizer=None,
                 **kwargs):
        '''

        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0
            (should be time).

        kernel_initializer: Initializer for the weights matrix
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the weights matrix
            (seeo [constraints](../constraints.md)).
        kwargs
        '''

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )

        super(AutoPool, self).__init__(**kwargs)

        self.axis = axis
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 3
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(1, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def get_config(self):
        config = {'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'axis': self.axis}

        base_config = super(AutoPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        scaled = self.kernel * x
        m = K.max(scaled, axis=self.axis, keepdims=True)
        sm = K.exp(scaled - m)
        w = sm / K.sum(sm, axis=self.axis, keepdims=True)
        return K.sum(x * w, axis=self.axis, keepdims=False)
