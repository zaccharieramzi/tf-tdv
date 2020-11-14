import numpy as np
import tensorflow as tf

from model import Model


def test_model():
    model = Model(
        layers_n_channels=[4, 8],
    )
    shape = [1, 32, 32, 1]
    res = model(tf.zeros(shape))
    assert res.shape.as_list() == shape


def test_model_change():
    model = Model(
        layers_n_channels=[4, 8],
    )
    x = tf.random.normal((1, 64, 64, 1))
    y = x
    model(x)
    before = [v.numpy() for v in model.trainable_variables]
    model.compile(optimizer='sgd', loss='mse')
    model.train_on_batch(x, y)
    after = [v.numpy() for v in model.trainable_variables]
    for b, a in zip(before, after):
        assert np.any(np.not_equal(b, a))
