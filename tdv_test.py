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
