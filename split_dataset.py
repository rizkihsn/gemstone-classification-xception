import os
import random
import shutil

# Agar hasil pembagian selalu sama
random.seed(42)

# Path dataset
train_dir = "dataset/train"
validation_dir = "dataset/validation"

# Persentase validation
validation_split = 0.2

# Buat folder validation jika belum ada
os.makedirs(validation_dir, exist_ok=True)

# Ambil semua folder kelas
classes = os.listdir(train_dir)

for class_name in classes:

    class_train_path = os.path.join(train_dir, class_name)
    class_validation_path = os.path.join(validation_dir, class_name)

    # Lewati jika bukan folder
    if not os.path.isdir(class_train_path):
        continue

    # Buat folder validation untuk kelas tersebut
    os.makedirs(class_validation_path, exist_ok=True)

    # Ambil semua gambar
    images = [
        img for img in os.listdir(class_train_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Acak urutan gambar
    random.shuffle(images)

    # Hitung jumlah validation
    val_size = int(len(images) * validation_split)

    validation_images = images[:val_size]

    # Pindahkan gambar ke validation
    for image in validation_images:

        src = os.path.join(class_train_path, image)
        dst = os.path.join(class_validation_path, image)

        shutil.move(src, dst)

print("======================================")
print(" Validation Dataset Berhasil Dibuat ")
print("======================================")