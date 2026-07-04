import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# KONFIGURASI
# =====================================================

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 10

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

MODEL_DIR = "model"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 50)
print("Konfigurasi berhasil dimuat.")
print("=" * 50)

# =====================================================
# DATA AUGMENTATION
# =====================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    zoom_range=0.30,
    width_shift_range=0.20,
    height_shift_range=0.20,
    shear_range=0.20,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# =====================================================
# LOAD DATASET
# =====================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

print("\nDataset berhasil dimuat.")
print(f"Jumlah kelas : {train_generator.num_classes}")
print(f"Jumlah data train : {train_generator.samples}")
print(f"Jumlah data validation : {validation_generator.samples}")
print(f"Jumlah data test : {test_generator.samples}")

print("\nLabel Kelas:")
print(train_generator.class_indices)

# =====================================================
# TRANSFER LEARNING XCEPTION
# =====================================================

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

# Freeze seluruh layer
base_model.trainable = False

print("\nBase Model Xception berhasil dimuat.")

# =====================================================
# CUSTOM CLASSIFICATION HEAD
# =====================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

from tensorflow.keras.regularizers import l2

x = Dense(
    256,
    activation="relu",
    kernel_regularizer=l2(0.001)
)(x)

x = Dropout(0.3)(x)

predictions = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

print("Custom classifier berhasil dibuat.")

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model berhasil di-compile.")

# =====================================================
# CALLBACKS
# =====================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "gemstone_xception.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

]

print("Callbacks berhasil dibuat.")

# =====================================================
# RINGKASAN MODEL
# =====================================================

print("\n")
print("=" * 50)
print("MODEL SUMMARY")
print("=" * 50)

model.summary()

print("\n")
print("=" * 50)
print("Semua persiapan selesai.")
print("Model siap untuk proses TRAINING.")
print("=" * 50)

# =====================================================
# TRAINING TAHAP 1 (FEATURE EXTRACTION)
# =====================================================

print("\n" + "=" * 60)
print("TRAINING TAHAP 1 : FEATURE EXTRACTION")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining Tahap 1 selesai.")

# =====================================================
# FINE TUNING
# =====================================================

print("\n" + "=" * 60)
print("FINE TUNING")
print("=" * 60)

base_model.trainable = True

# Unfreeze semua layer kecuali 50 layer terakhir (memperkuat fine-tuning)
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 15

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS + fine_tune_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=callbacks
)

print("\nFine Tuning selesai.")

# =====================================================
# EVALUASI MODEL
# =====================================================

print("\n" + "=" * 60)
print("EVALUASI MODEL")
print("=" * 60)

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")

# =====================================================
# SIMPAN MODEL
# =====================================================

model.save("model/gemstone_xception_final.keras")

print("\nModel berhasil disimpan.")

# =====================================================
# SIMPAN HISTORY
# =====================================================

history_df = pd.DataFrame(history.history)
history_df.to_csv("results/history.csv", index=False)

print("History training berhasil disimpan.")

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("results/accuracy.png")
plt.close()

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/loss.png")
plt.close()

print("Grafik berhasil disimpan.")