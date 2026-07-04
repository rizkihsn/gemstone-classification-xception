from predict import predict_image

image_path = "dataset/test/Diamond/diamond_3.jpg"

label, confidence = predict_image(image_path)

print("============================")
print("HASIL PREDIKSI")
print("============================")
print("Jenis Batu :", label)
print(f"Confidence : {confidence:.2f}%")