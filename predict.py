import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import gdown

# =====================================================
# AUTO-DOWNLOAD MODEL DARI GOOGLE DRIVE
# =====================================================
MODEL_PATH = "model/gemstone_xception_final.keras"

# Ganti GDRIVE_FILE_ID dengan File ID dari link share Google Drive Anda
# Contoh link: https://drive.google.com/file/d/1A2B3C4D5E6F.../view?usp=sharing
#                                                ^^^^^^^^^^^^^^^^^ ini adalah File ID-nya
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "1HwfegIGfP-WkmQQLusLNAsUEhrtW1O5I")

if not os.path.exists(MODEL_PATH):
    print(">>> Model tidak ditemukan. Mengunduh dari Google Drive...")
    os.makedirs("model", exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print(">>> Download selesai!")
else:
    print(">>> Model ditemukan secara lokal, skip download.")

# =====================================================
# FIX UNTUK ERROR GLOROTUNIFORM VERSION MISMATCH
# =====================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class CompatibleGlorotUniform(tf.keras.initializers.GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop('input_axes', None)
        kwargs.pop('output_axes', None)
        super().__init__(seed=seed, **kwargs)

# Load model
model = load_model(
    MODEL_PATH,
    custom_objects={"GlorotUniform": CompatibleGlorotUniform}
)
print(">>> Model berhasil dimuat!")
# =====================================================


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