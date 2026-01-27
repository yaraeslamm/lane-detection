import tensorflow as tf


class MeanIoUFromLogits(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def lane_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.squeeze(y_true, axis=-1)

    lane_true = tf.cast(y_true == 2, tf.float32)
    lane_pred = tf.cast(y_pred == 2, tf.float32)

    intersection = tf.reduce_sum(lane_true * lane_pred)
    union = tf.reduce_sum(lane_true) + tf.reduce_sum(lane_pred) - intersection

    return intersection / (union + 1e-6)
