import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from dataset import make_dataset
from metrics import MeanIoUFromLogits, lane_iou

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 10

PRETRAINED_MODEL_PATH = "road_model_industryV1_myDATASET.keras"
OUTPUT_MODEL_PATH = "road_model_finetuned.keras"

# -----------------------------
# LOAD DATA
# -----------------------------
# These lists come from however YOU load paths
# (Kaggle paths, local paths, etc.)
# Example:
# train_images = [...]
# train_masks  = [...]
# val_images   = [...]
# val_masks    = [...]

train_ds = make_dataset(
    train_images,
    train_masks,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = make_dataset(
    val_images,
    val_masks,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# LOAD PRETRAINED MODEL
# -----------------------------
model = tf.keras.models.load_model(
    PRETRAINED_MODEL_PATH,
    custom_objects={
        "MeanIoUFromLogits": MeanIoUFromLogits,
        "lane_iou": lane_iou
    }
)

# OPTIONAL: freeze early layers if you want
# for layer in model.layers[:10]:
#     layer.trainable = False

# -----------------------------
# COMPILE (LOW LR FOR FINETUNING)
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        MeanIoUFromLogits(num_classes=3, name="miou"),
        lane_iou
    ]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
checkpoint = ModelCheckpoint(
    OUTPUT_MODEL_PATH,
    monitor="val_miou",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_miou",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)
