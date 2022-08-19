import tensorflow as tf


def getTFDS(data_dir, img_size, batch_size, colormode='rgb'):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                   color_mode=colormode,
                                                                   image_size=(img_size, img_size),
                                                                   batch_size=batch_size)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds


def getTFDS_split(data_dir, img_size, batch_size, colormode='rgb', split_rate=0.2):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                   validation_split=split_rate,
                                                                   subset="training",
                                                                   seed=123,
                                                                   color_mode=colormode,
                                                                   image_size=(img_size, img_size),
                                                                   batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                 validation_split=split_rate,
                                                                 subset="validation",
                                                                 seed=123,
                                                                 color_mode=colormode,
                                                                 image_size=(img_size, img_size),
                                                                 batch_size=batch_size)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds
