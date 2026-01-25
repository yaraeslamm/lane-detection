# src/inference.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


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


class RoadSegmentationModel:
    def __init__(self, model_path, input_size=(256, 256)):
        self.input_size = input_size
        self.model = load_model(
            model_path,
            custom_objects={
                "lane_iou": lane_iou,
                "MeanIoUFromLogits": MeanIoUFromLogits,
            }
        )

    def predict(self, image):
        """
        Returns:
            class_mask (H, W) with values {0,1,2}
        """
        h, w = image.shape[:2]

        img = cv2.resize(image, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0]  # (256,256,3)
        class_mask = np.argmax(pred, axis=-1)

        class_mask = cv2.resize(
            class_mask, (w, h), interpolation=cv2.INTER_NEAREST
        )

        return class_mask
