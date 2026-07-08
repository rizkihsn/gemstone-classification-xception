import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# --- FIX UNTUK ERROR GLOROTUNIFORM VERSION MISMATCH ---
# Membuat initializer modifikasi agar mengabaikan input_axes dan output_axes jika tidak didukung
@tf.keras.utils.register_keras_serializable(package="Custom")
class CompatibleGlorotUniform(tf.keras.initializers.GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop('input_axes', None)
        kwargs.pop('output_axes', None)
        super().__init__(seed=seed, **kwargs)

# Load model dengan menyisipkan CompatibleGlorotUniform ke custom_objects
model = load_model(
    "model/gemstone_xception_final.keras",
    custom_objects={"GlorotUniform": CompatibleGlorotUniform}
)
# -----------------------------------------------------

# Label kelas (HARUS sama dengan train_generator.class_indices)
CLASS_NAMES = [
    "Amethyst",
    "Aquamarine",
    "Citrine",
    "Diamond",
    "Emerald",
    "Peridot",
    "Ruby",
    "Sapphire Blue",
    "Tigers Eye",
    "Tourmaline"
]

def predict_image(image_path):
    # Buka gambar dan konversi ke RGB
    image = Image.open(image_path).convert("RGB")
    
    # Pad gambar menjadi persegi agar aspect ratio tidak rusak (meningkatkan akurasi algoritma)
    image = ImageOps.pad(image, (299, 299), color=(0, 0, 0))
    
    # Konversi ke array numpy
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    
    # Preprocessing Xception
    image = preprocess_input(image)

    # Lakukan prediksi
    prediction = model.predict(image, verbose=0)

    # Ambil index dan confidence tertinggi
    index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return CLASS_NAMES[index], confidence