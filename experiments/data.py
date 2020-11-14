"""Mostly inspired by
https://github.com/zaccharieramzi/understanding-unets/blob/master/learning_wavelets/data/datasets.py
"""
from collections.abc import Iterable
from functools import partial

import tensorflow as tf

from config import *


# normalisation
def normalise(image):
    return (image / 255) - 0.5

# patch selection
def select_patch_in_image(image, patch_size=96, seed=0):
    if patch_size is not None:
        patch = tf.image.random_crop(
            image,
            [patch_size, patch_size, 1],
            seed=seed,
        )
        return patch
    else:
        return image

# noise
def add_noise(image, noise_std_range=(0, 50)):
    if not isinstance(noise_std_range, Iterable):
        noise_std_range = (noise_std_range, noise_std_range)
    noise_std = tf.random.uniform(
        (1,),
        minval=noise_std_range[0],
        maxval=noise_std_range[1],
    )
    noise = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=noise_std/255,
        dtype=tf.float32,
    )
    return image + noise

def im_dataset_bsd500(mode='training', **kwargs):
    # the training set for bsd500 is test + train
    # the test set (i.e. containing bsd68 images) is val
    if mode == 'training':
        train_path = 'BSR/BSDS500/data/images/train'
        test_path = 'BSR/BSDS500/data/images/test'
        paths = [train_path, test_path]
    elif mode == 'validation' or mode == 'testing':
        val_path = 'BSR/BSDS500/data/images/val'
        paths = [val_path]
    im_ds = im_dataset(BSD500_DATA_DIR, paths, 'jpg', from_rgb=True, **kwargs)
    return im_ds

def im_dataset_bsd68(grey=True, **kwargs):
    if not grey:
        raise ValueError('Color images not available for BSD68')
    path = 'BSD68'
    im_ds = im_dataset(BSD68_DATA_DIR, [path], 'png', **kwargs)
    return im_ds

def im_dataset(
        data_dir,
        paths,
        pattern,
        batch_size=32,
        patch_size=96,
        noise_std=(0, 50),
        n_samples=None,
        from_rgb=False,
        grey=True,
    ):
    file_ds = None
    for path in paths:
        file_ds_new = tf.data.Dataset.list_files(f'{data_dir}{path}/*.{pattern}', shuffle=False)
        if file_ds is None:
            file_ds = file_ds_new
        else:
            file_ds.concatenate(file_ds_new)
    file_ds = file_ds.shuffle(800, seed=0, reshuffle_each_iteration=False)
    if n_samples is not None:
        file_ds = file_ds.take(n_samples)
    if pattern == 'jpg':
        decode_function = tf.image.decode_jpeg
    elif pattern == 'png':
        decode_function = tf.image.decode_png
    image_ds = file_ds.map(
        tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).map(
        decode_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if from_rgb and grey:
        image_ds = image_ds.map(
            tf.image.rgb_to_grayscale, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    image_grey_ds = image_ds.map(
        normalise, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # TODO: add data augmentation
    if patch_size is not None:
        image_patch_ds = image_grey_ds.map(
            partial(select_patch_in_image, patch_size=patch_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    image_noisy_ds = image_patch_ds.map(
        lambda patch: (partial(add_noise, noise_std_range=noise_std)(patch), patch),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    image_noisy_ds = image_noisy_ds.batch(batch_size)
    image_noisy_ds = image_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_noisy_ds
