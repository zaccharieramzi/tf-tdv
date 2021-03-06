import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv2D, Activation, MaxPool2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model


BLUR_KERNEL = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
BLUR_KERNEL = BLUR_KERNEL @ BLUR_KERNEL.T
BLUR_KERNEL /= BLUR_KERNEL.sum()

class Conv(Layer):
    def __init__(self, n_filters=32, kernel_size=3, use_bias=False, kernel_constraint=None, strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_constraint = kernel_constraint
        self.strides = strides
        self.pad_size = self.kernel_size // 2
        self.paddings = [
            [0, 0],
            [self.pad_size, self.pad_size],
            [self.pad_size, self.pad_size],
            [0, 0],
        ]
        self.conv = Conv2D(
            self.n_filters,
            self.kernel_size,
            padding='valid',
            strides=self.strides,
            use_bias=self.use_bias,
            kernel_constraint=self.kernel_constraint,
        )

    def call(self, inputs):
        outputs = tf.pad(
            inputs,
            self.paddings,
            "SYMMETRIC",
        )
        outputs = self.conv(outputs)
        return outputs

class ConvTranspose(Layer):
    def __init__(self, n_filters=32, kernel_size=3, use_bias=False, kernel_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_constraint = kernel_constraint
        self.conv = Conv2DTranspose(
            self.n_filters,
            self.kernel_size,
            strides=(2, 2),
            padding='same',
            use_bias=self.use_bias,
            kernel_constraint=self.kernel_constraint,
        )

    def call(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class StudentActivation(Layer):
    # I found out in the code that nu is actually 1. and not 9. like stated
    # in the paper
    # https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py#L41
    def __init__(self, nu=1, **kwargs):
        super().__init__(**kwargs)
        if nu is None:
            self.nu = self.add_weight(initializer=tf.initializers.constant(1.))
        else:
            self.nu = nu

    def call(self, inputs):
        outputs = (1/(2*self.nu)) * tf.math.log(1 + self.nu * inputs**2)
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
        self.last_conv = Conv(
            self.n_filters,
            kernel_size=3,
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
        if self.pooling == 'conv':
            self.pool = Conv(n_filters, 3, use_bias=False)
        else:
            self.pool = MaxPool2D((2, 2), strides=(1, 1))

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.blur_kernel = tf.tile(self.blur_kernel, [1, 1, in_channels, 1])
        self.point_wise_kernel = tf.eye(in_channels)[None, None]

    def call(self, inputs):
        padded_inputs = tf.pad(
            inputs,
            [
                [0, 0],
                [2, 2],
                [2, 2],
                [0, 0],
            ],
            "SYMMETRIC",
        )
        blurred_downsample_inputs = tf.nn.separable_conv2d(
            padded_inputs,
            self.blur_kernel,
            self.point_wise_kernel,
            strides=[1, 2, 2, 1],
            padding='VALID',
        )
        outputs = self.pool(blurred_downsample_inputs)
        return outputs

class BlurUpSample(Layer):
    def __init__(self, unpooling='conv', n_filters=32, **kwargs):
        super().__init__(**kwargs)
        self.unpooling = unpooling
        self.blur_kernel = tf.constant(BLUR_KERNEL)[..., None, None]
        if self.unpooling == 'conv':
            self.unpool = ConvTranspose(n_filters, 3, use_bias=False)
        else:
            self.unpool = UpSampling2D((2, 2))

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.blur_kernel = tf.tile(self.blur_kernel, [1, 1, in_channels, 1])
        self.point_wise_kernel = tf.eye(in_channels)[None, None]

    def call(self, inputs):
        outputs = self.unpool(inputs)
        outputs = tf.pad(
            outputs,
            [
                [0, 0],
                [2, 2],
                [2, 2],
                [0, 0],
            ],
            "SYMMETRIC",
        )
        outputs = tf.nn.separable_conv2d(
            outputs,
            self.blur_kernel,
            self.point_wise_kernel,
            strides=[1, 1, 1, 1],
            padding='VALID',
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
                    unpooling=pooling,
                    n_filters=self.n_filters*self.multiplier**i_scale,
                )
                for i_scale in range(self.n_scales-1)
            ]
        elif self.pooling == 'conv':
            self.pools = [
                Conv(
                    self.n_filters*self.multiplier**i_scale,
                    kernel_size=3,
                    strides=2,
                )
                for i_scale in range(self.n_scales)
            ]
            self.unpools = [
                ConvTranspose(
                    self.n_filters*self.multiplier**i_scale,
                    kernel_size=3,
                )
                for i_scale in range(self.n_scales-1)
            ]
        else:
            raise ValueError(f'Not possible to using pooling {self.pooling}')
        # concatenation
        self.conv_concats = [
            Conv(
                self.n_filters*self.multiplier**i_scale,
                kernel_size=1,
                use_bias=False,
            )
            for i_scale in range(self.n_scales-1)
        ]

    def call(self, inputs):
        scales = []
        if self.first_macro:
            outputs = inputs
        else:
            outputs = inputs[0]
        for i_scale in range(self.n_scales):
            if self.first_macro:
                res_outputs = None
            else:
                res_outputs = inputs[i_scale]
            if i_scale > 0:
                if res_outputs is not None:
                    outputs = outputs + res_outputs
            outputs = self.down_blocks[i_scale](outputs)
            if i_scale < self.n_scales - 1:
                scales.append(outputs)
                outputs =  self.pools[i_scale](outputs)
        all_outputs = [outputs]
        for i_scale in range(self.n_scales - 2, -1, -1):
            outputs = self.unpools[i_scale](outputs)
            outputs = tf.concat([outputs, scales[i_scale]], axis=-1)
            outputs = self.conv_concats[i_scale](outputs)
            outputs = self.up_blocks[i_scale](outputs)
            all_outputs.append(outputs)
        if self.last_macro:
            return outputs
        else:
            return all_outputs[::-1]

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
        self.macro_blocks = [
            MacroBlock(
                n_scales=self.n_scales,
                n_filters=self.n_filters,
                multiplier=self.multiplier,
                activation_str=self.activation_str,
                pooling=self.pooling,
                use_bias=self.use_bias,
                first_macro=i_macro==0,
                last_macro=i_macro==self.n_macro-1,
            )
            for i_macro in range(self.n_macro)
        ]

    def call(self, inputs):
        outputs = inputs
        for block in self.macro_blocks:
            outputs = block(outputs)
        return outputs

class ZeroMean(Constraint):
    def __call__(self, w):
        current_mean = tf.reduce_mean(w, axis=[0, 1, 2])
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
            shallow=False,
            denoising=False,
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
        self.shallow = shallow
        self.denoising = denoising
        if self.denoising:
            self.alpha = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.constant(0.01),
            )
        self.K = Conv(
            self.n_filters,
            kernel_size=3,  # got this from the code
            # https://github.com/VLOGroup/tdv/blob/fe220b3c39/ddr/tdv.py#L186
            use_bias=False,
            kernel_constraint=ZeroMean(),
        )
        if self.shallow:
            self.N = tf.abs
        else:
            self.N = UnetMultiscaleResidual(
                n_macro=self.n_macro,
                n_scales=self.n_scales,
                n_filters=self.n_filters,
                multiplier=self.multiplier,
                pooling=self.pooling,
                activation_str=self.activation_str,
                use_bias=self.use_bias,
            )
        self.w = Conv(
            1,
            kernel_size=1,
            use_bias=False,
        )


    def call(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(inputs)
            r = self.energy(inputs)
        grad = g.gradient(r, inputs)
        if self.denoising:
            outputs = inputs - self.alpha * grad
        else:
            outputs = grad
        return outputs

    def energy(self, inputs):
        high_pass_inputs = self.K(inputs)
        outputs = self.N(high_pass_inputs)
        outputs = self.w(outputs)
        outputs = tf.reduce_sum(outputs, axis=[1, 2])
        return outputs

class TV(Model):
    def call(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(inputs)
            r = self.energy(inputs)
        grad = g.gradient(r, inputs)
        return grad

    def energy(self, inputs):
        outputs = tf.image.total_variation(inputs)
        return outputs
