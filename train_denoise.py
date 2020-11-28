from tensorflow_addons.callbacks import TQDMProgressBar
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from experiments.data import im_dataset_bsd500
from experiments.unrolled_fb import UnrolledFB
from tdv import TDV


def train(batch_size=8, noise_std=(25,25), n_steps=1, model_kwargs=None, non_linearity='student'):
    train_ds = im_dataset_bsd500(batch_size=batch_size, noise_std=noise_std)
    if model_kwargs is None:
        model_kwargs = dict(
            model_class=TDV,
            model_kwargs={'activation_str': non_linearity},
            init_step_size=0.0001,
            n_iter=10,
        )
    model = UnrolledFB(**model_kwargs)
    model.compile(loss='mse', optimizer=Adam(1e-4))
    n_steps_per_epoch = 400 // batch_size
    epochs = int(n_steps//n_steps_per_epoch)
    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=n_steps_per_epoch,
        callbacks=[TQDMProgressBar(), ModelCheckpoint(save_weights_only=True, period=epochs)],
        verbose=0,
    )
