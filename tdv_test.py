import numpy as np
import pytest
import tensorflow as tf

from tdv import TDV


def test_tdv_energy():
    model = TDV(
        n_macro=2,
        n_scales=2,
        n_filters=8,
    )
    shape = [1, 32, 32, 1]
    res = model.energy(tf.zeros(shape))
    assert res.shape.as_list() == [1, 1]

@pytest.mark.parametrize('n_macro', [2, 3])
@pytest.mark.parametrize('n_scales', [2, 3])
def test_tdv_call(n_macro, n_scales):
    model = TDV(
        n_macro=n_macro,
        n_scales=n_scales,
        n_filters=8,
    )
    shape = [1, 32, 32, 1]
    res = model(tf.zeros(shape))
    assert res.shape.as_list() == shape


def test_tdv_change():
    model = TDV(
        n_macro=2,
        n_scales=2,
        n_filters=8,
    )
    x = tf.random.normal((1, 32, 32, 1))
    y = tf.random.normal((1, 32, 32, 1))
    model(x)
    before = [v.numpy() for v in model.trainable_variables]
    model.compile(optimizer='sgd', loss='mse')
    model.train_on_batch(x, y)
    after = [v.numpy() for v in model.trainable_variables]
    for b, a in zip(before, after):
        assert np.any(np.not_equal(b, a))
        assert not np.any(np.isnan(b))

@pytest.mark.parametrize('n_macro', [2, 3])
@pytest.mark.parametrize('n_scales', [2, 3])
def test_tdv_gradient(n_macro, n_scales):
    # this is more of a test that tensorflow does the correct thing
    # when applying the gradient tape since this should be taken into account
    # but since I heard some ppl telling there might be silent bugs, I prefer
    # checking
    # setup the data
    x = tf.random.normal((1, 32, 32, 1))

    # define the TDV regularizer
    model = TDV(
        n_macro=n_macro,
        n_scales=n_scales,
        n_filters=8,
    )

    def compute_loss(scale):
        return model.energy(scale*x)

    scale = 1.

    # compute the gradient using the implementation
    grad_scale = tf.reduce_sum(x*model(scale*x))

    # check it numerically
    epsilon = 1e-4
    l_p = compute_loss(scale+epsilon).item()
    l_n = compute_loss(scale-epsilon).item()
    grad_scale_num = (l_p - l_n) / (2 * epsilon)

    condition = np.abs(grad_scale - grad_scale_num) < 1e-3
    print(f'grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
    assert condition
