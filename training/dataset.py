import tensorflow as tf

IMG_SIZE = (256, 256)

def load_image_mask(img_path, mask_path):
    """
    Loads an image and its segmentation mask and converts them
    into tensors suitable for training.
    """

    # ---- Image ----
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    # ---- Mask ----
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")

    r = mask[:, :, 0]
    g = mask[:, :, 1]

    background = tf.cast((r == 0) & (g == 0), tf.uint8)
    road       = tf.cast((r == 128) & (g == 0), tf.uint8)
    lane       = tf.cast((r == 0) & (g >= 128), tf.uint8)

    class_mask = background * 0 + road * 1 + lane * 2
    class_mask = tf.expand_dims(class_mask, axis=-1)

    return img, class_mask


def make_dataset(image_paths, mask_paths, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        ds = ds.shuffle(100)

    ds = ds.map(
        load_image_mask,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
