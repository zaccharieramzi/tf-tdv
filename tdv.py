import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv2D, Activation, MaxPool2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model


BLUR_KERNEL = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
BLUR_KERNEL = BLUR_KERNEL @ BLUR_KERNEL.T
BLUR_KERNEL /= BLUR_KERNEL.sum()

class StudentActivation(Layer):
    def __init__(self, nu=9, **kwargs):
        super().__init__(**kwargs)
        if nu is None:
            self.nu = self.add_weight(initializer=tf.initializers.constant(1.))
        else:
            self.nu = nu

    def call(self, inputs):
        outputs = (1/2*self.nu) * tf.log(1 + self.nu * inputs**2)
        return outputs

class MicroBlock(Layer):
    def __init__(self, n_filters=32, activation_str='student', use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.use_bias = use_bias
        self.first_conv = Conv2D(
            self.n_filters,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=self.use_bias,
        )
        self.activation_str = activation_str
        if self.activation_str == 'student':
            self.activation = StudentActivation()
        else:
            self.activation = Activation(self.activation_str)
        self.last_conv = Conv2D(
            self.n_filters,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=self.use_bias,
        )

    # TODO: maybe code a projection conv 1x1 layer in order to handle
    # different input and output sizes

    def call(self, inputs):
        outputs = self.first_conv(inputs)
        outputs = self.activation(outputs)
        outputs = self.last_conv(outputs)
        ouputs = inputs + outputs
        return ouputs

class BlurDownSample(Layer):
    def __init__(self, pooling='conv', n_filters=32, **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling
        self.blur_kernel = tf.constant(BLUR_KERNEL)[..., None, None]
        self.point_wise_kernel = tf.constant(1.)[None, None, None, None]
        if self.pooling == 'conv':
            self.pool = Conv2D(n_filters, 3, padding='same', use_bias=False)
        else:
            self.pool = MaxPool2D((2, 2), strides=(1, 1))

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.blur_kernel = tf.tile(self.blur_kernel, [1, 1, in_channels, 1])
        self.point_wise_kernel = tf.tile(self.point_wise_kernel, [1, 1, in_channels, in_channels])

    def call(self, inputs):
        blurred_downsample_inputs = tf.nn.separable_conv2d(
            inputs,
            self.blur_kernel,
            self.point_wise_kernel,
            strides=2,
            padding='same',
        )
        outputs = self.pool(blurred_downsample_inputs)
        return outputs

class BlurUpSample(Layer):
    def __init__(self, unpooling='conv', n_filters=32, **kwargs):
        super().__init__(**kwargs)
        self.unpooling = unpooling
        self.blur_kernel = tf.constant(BLUR_KERNEL)[..., None, None]
        self.point_wise_kernel = tf.constant(1.)[None, None, None, None]
        if self.unpooling == 'conv':
            self.unpool = Conv2DTranspose(n_filters, 3, strides=(2, 2), padding='same', use_bias=False)
        else:
            self.unpool = UpSampling2D((2, 2))

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.blur_kernel = tf.tile(self.blur_kernel, [1, 1, in_channels, 1])
        self.point_wise_kernel = tf.tile(self.point_wise_kernel, [1, 1, in_channels, in_channels])

    def call(self, inputs):
        outputs = self.unpool(inputs)
        outputs = tf.nn.separable_conv2d(
            outputs,
            self.blur_kernel,
            self.point_wise_kernel,
            strides=1,
            padding='same',
        )
        return outputs

class MacroBlock(Layer):
    def __init__(
            self,
            n_scales=3,
            n_filters=32,
            multiplier=1,
            activation_str='student',
            pooling='blur-conv',
            use_bias=False,
            first_macro=False,
            last_macro=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_scales = n_scales
        self.n_filters = n_filters
        self.multiplier = multiplier
        self.activation_str = activation_str
        self.pooling = pooling
        self.use_bias = use_bias
        self.first_macro = first_macro
        self.last_macro = last_macro
        # micro blocks
        self.down_blocks = [
            MicroBlock(
                n_filters=self.n_filters*self.multiplier**i_scale,
                activation_str=self.activation_str,
                use_bias=self.use_bias,
            )
            for i_scale in range(self.n_scales)
        ]
        self.up_blocks = [
            MicroBlock(
                n_filters=self.n_filters*self.multiplier**i_scale,
                activation_str=self.activation_str,
                use_bias=self.use_bias,
            )
            for i_scale in range(self.n_scales-1)
        ]
        # down/up-sampling
        if 'blur' in self.pooling:
            pooling = self.pooling.split('-')[-1]
            self.pools = [
                BlurDownSample(
                    pooling=pooling,
                    n_filters=self.n_filters*self.multiplier**i_scale,
                )
                for i_scale in range(self.n_scales)
            ]
            self.unpools = [
                BlurUpSample(
                    pooling=pooling,
                    n_filters=self.n_filters*self.multiplier**i_scale,
                )
                for i_scale in range(self.n_scales-1)
            ]
        elif self.pooling == 'conv':
            self.pools = [
                Conv2D(
                    self.n_filters*self.multiplier**i_scale,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                )
                for i_scale in range(self.n_scales)
            ]
            self.unpools = [
                Conv2DTranspose(
                    self.n_filters*self.multiplier**i_scale,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                )
                for i_scale in range(self.n_scales-1)
            ]
        else:
            self.pools = [MaxPool2D((2, 2)) for i_scale in range(self.n_scales)]
            self.unpools = [UpSampling2D((2, 2)) for i_scale in range(self.n_scales-1)]
        # concatenation
        self.conv_concats = [
            Conv2D(
                self.n_filters*self.multiplier**i_scale,
                kernel_size=1,
                padding='same',
                use_bias=False,
            )
            for i_scale in range(self.n_scales-1)
        ]

    def call(self, inputs):
        scales = []
        outputs = None
        for i_scale in range(self.n_scales):
            if self.first_macro:
                res_outputs = inputs
            else:
                res_outputs = inputs[0]
            if outputs is None:
                outputs = res_outputs
            else:
                outputs = res_outputs + outputs
            outputs = self.down_blocks[i_scale](outputs)
            if i_scale < self.n_scales - 1:
                scales.append(outputs)
                outputs =  self.pools[i_scale](outputs)
        all_outputs = [outputs]
        for i_scale in range(self.n_scales - 2, 0, -1):
            outputs = self.unpools[i_scale](outputs)
            outputs = tf.concatenate([outputs, scales[i_scale]], axis=-1)
            outputs = self.conv_concats[i_scale](outputs)
            outputs = self.up_blocks[i_scale](outputs)
            all_outputs.append(outputs)
        if self.last_macro:
            return outputs
        else:
            return all_outputs

class UnetMultiscaleResidual(Model):
    def __init__(
            self,
            n_macro=3,
            n_scales=3,
            n_filters=32,
            multiplier=1,
            pooling='blur-conv',
            activation_str='student',
            use_bias=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_macro = n_macro
        self.n_scales = n_scales
        self.n_filters = n_filters
        self.multiplier = multiplier
        self.pooling = pooling
        self.activation_str = activation_str
        self.use_bias = use_bias
        macro_blocks_kwargs = dict(
            n_scales=self.n_scales,
            n_filters=self.n_filters,
            multiplier=self.multiplier,
            activation_str=self.activation_str,
            pooling=self.pooling,
            use_bias=self.use_bias,
        )
        self.first_macro_block = MacroBlock(
            first_macro=True,
            name='first_macro_block',
            **macro_blocks_kwargs,
        )
        self.macro_blocks = [MacroBlock(**macro_blocks_kwargs) for _ in range(self.n_macro-2)]
        self.last_macro_block = MacroBlock(
            last_macro=True,
            name='last_macro_block',
            **macro_blocks_kwargs,
        )

    def call(self, inputs):
        outputs = self.first_macro_block(inputs)
        for block in self.macro_blocks:
            outputs = block(outputs)
        outputs = self.last_macro_block(outputs)
        return outputs

class ZeroMean(Constraint):
    def __call__(self, w):
        current_mean = tf.reduce_mean(w, axis=[1, 2, 3])
        w = w - current_mean
        return w

class TDV(Model):
    def __init__(
            self,
            n_macro=3,
            n_scales=3,
            n_filters=32,
            multiplier=1,
            pooling='blur-conv',
            activation_str='student',
            use_bias=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_macro = n_macro
        self.n_scales = n_scales
        self.n_filters = n_filters
        self.multiplier = multiplier
        self.pooling = pooling
        self.activation_str = activation_str
        self.use_bias = use_bias
        self.K = Conv2D(
            self.n_filters,
            kernel_size=3,  # got this from the code
            # https://github.com/VLOGroup/tdv/blob/fe220b3c39/ddr/tdv.py#L186
            padding='same',
            use_bias=False,
            kernel_constraint=ZeroMean(),
        )
        self.N = UnetMultiscaleResidual(
            n_macro=self.n_macro,
            n_scales=self.n_scales,
            n_filters=self.n_filters,
            multiplier=self.multiplier,
            pooling=self.pooling,
            activation_str=self.activation_str,
            use_bias=self.use_bias,
        )
        self.w = Conv2D(
            1,
            kernel_size=1,
            padding='same',
            use_bias=False,
        )


    def call(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(inputs)
            r = self.energy(inputs)
        prox = g.gradient(r, inputs)
        return prox

    def energy(self, inputs):
        high_pass_inputs = self.K(inputs)
        outputs = self.N(high_pass_inputs)
        outputs = self.w(outputs)
        outputs = tf.reduce_sum(outputs, axis=[1, 2])
        return outputs
