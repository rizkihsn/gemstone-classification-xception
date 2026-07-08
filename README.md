# Gemstone Classification AI 💎

![Gemstone Classification](static/diamond_hero.png)

A modern, web-based Artificial Intelligence application that can instantly classify 10 different types of gemstones using a Deep Learning model powered by Transfer Learning (Xception). 

This project is built with **Flask (Python)** for the backend and a premium UI/UX frontend utilizing **Bootstrap 5**, Custom CSS Glassmorphism, and responsive design.

## Features ✨

- **High Accuracy Model**: Achieves up to 98.5% validation accuracy using the Xception CNN architecture.
- **Auto-Download Model**: The `.keras` model is too large for GitHub (>100MB), so the app is configured to automatically download the trained model from Google Drive during startup/deployment.
- **Interactive UI/UX**: Drag-and-drop file upload, instant image preview, custom dark mode, and sleek animations.
- **Fast Inference**: Image preprocessing (aspect-ratio preserving padding) and model prediction happen in under 2 seconds.
- **Production Ready**: Configured with `gunicorn` for easy deployment to Railway, Render, or Heroku.

## Supported Gemstones

The model is currently trained to classify these 10 gemstones:
- Amethyst
- Aquamarine
- Citrine
- Diamond
- Emerald
- Peridot
- Ruby
- Sapphire Blue
- Tigers Eye
- Tourmaline

## Tech Stack 🛠️

- **Backend**: Python 3, Flask, Werkzeug, Gunicorn
- **Machine Learning**: TensorFlow 2.x, Keras, NumPy, Pillow, Pandas
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Bootstrap 5, FontAwesome, Bootstrap Icons
- **Deployment Utility**: gdown (for auto-downloading weights from GDrive)

## Installation & Setup 🚀

### 1. Clone the repository
```bash
git clone https://github.com/rizkihsn/gemstone-classification-xception.git
cd gemstone-classification-xception
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application Locally
```bash
python app.py
```
*Note: On the first run, the app will automatically download the ~150MB model file from Google Drive to the `/model` directory. Please make sure you have an active internet connection.*

### 5. Access the Web App
Open your browser and navigate to:
`http://localhost:8080` (or the port shown in your terminal)

## How to Deploy to Railway / Render ☁️

1. Upload/Push this code to your GitHub repository.
2. Link your GitHub repository to your Railway/Render account.
3. The platform will automatically detect `requirements.txt` and `Procfile`.
4. It will install all dependencies, start `gunicorn`, auto-download the model from Google Drive, and serve the application online!

## Training the Model 🧠
If you wish to retrain the model with your own dataset or tweak the hyperparameters:
1. Place your dataset inside the `dataset/train`, `dataset/validation`, and `dataset/test` folders.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The new model will be saved as `model/gemstone_xception_final.keras`.

## Credits
Built as a Capstone Project exploring the capabilities of Convolutional Neural Networks and Transfer Learning for Image Classification.
