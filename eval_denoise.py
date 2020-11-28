import tensorflow as tf

from experiments.data import im_dataset_bsd500
from experiments.unrolled_fb import UnrolledFB
from tdv import TDV


def pad_for_pool(inputs, n_pools):
    problematic_dims = tf.shape(inputs)[1:3]
    k = tf.math.floordiv(problematic_dims, 2 ** n_pools)
    n_pad = tf.where(
        tf.math.mod(problematic_dims, 2 ** n_pools) == 0,
        0,
        (k + 1) * 2 ** n_pools - problematic_dims,
    )
    left_padding = tf.where(
        tf.logical_or(tf.math.mod(problematic_dims, 2) == 0, n_pad == 0),
        n_pad//2,
        n_pad//2 + 1,
    )
    right_padding = n_pad//2
    paddings = [
        (0, 0),
        (left_padding[0], right_padding[0]),
        (left_padding[1], right_padding[1]),
        (0, 0),
    ]
    inputs_padded = tf.pad(inputs, paddings)
    return inputs_padded, paddings

class MultiScaleModel(tf.keras.models.Model):
    def __init__(self, model, n_scales=0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.n_scales = n_scales

    def call(self, inputs):
        if self.n_scales > 0:
            outputs, paddings = pad_for_pool(inputs, n_pools=self.n_scales)
        else:
            outputs = inputs
        outputs = self.model(outputs)
        if self.n_scales > 0:
            problematic_dims = tf.shape(outputs)[1:3]
            outputs = outputs[
                :,
                paddings[1][0]: problematic_dims[0] - paddings[1][1],
                paddings[2][0]: problematic_dims[1] - paddings[2][1],
                :,
            ]
        return outputs

def tf_psnr(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)

def eval(batch_size=1, noise_std=(25,25), n_samples=100, model_kwargs=None, non_linearity='student'):
    val_ds = im_dataset_bsd500(
        mode='validation',
        batch_size=batch_size,
        noise_std=noise_std,
        patch_size=None,
    )
    if model_kwargs is None:
        model_kwargs = dict(
            model_class=TDV,
            model_kwargs={'activation_str': non_linearity},
            init_step_size=0.0001,
            n_iter=10,
        )
    model = UnrolledFB(**model_kwargs)
    model(tf.ones([1, 32, 32, 1]))
    model.load_weights(f'denoising_unrolled_fb_tdv_{non_linearity}.h5')
    full_model = MultiScaleModel(model, n_scales=4)
    full_model.compile(loss='mse', metrics=[tf_psnr])
    eval_res = full_model.evaluate(val_ds.take(n_samples))
    print(eval_res)
    return eval_res
