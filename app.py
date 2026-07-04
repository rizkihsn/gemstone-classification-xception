import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if "image" not in request.files:
        return render_template(
            "result.html",
            prediction="Tidak ada gambar.",
            confidence=0,
            image=None
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "result.html",
            prediction="Belum memilih gambar.",
            confidence=0,
            image=None
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    prediction, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        image=filename
    )


if __name__ == "__main__":
    app.run(debug=True)