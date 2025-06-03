from flask import Flask, request, send_file, jsonify
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image
import numpy as np
import io

# === Register custom loss ===
@register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.7):
    smooth = 1e-6
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

# === Load model ===
MODEL_PATH = "mobilenet_mammo.keras"  # Adjust this if needed
model = keras.models.load_model(MODEL_PATH, compile=False)

# === Image size ===
IMAGE_SIZE = (256, 256)  # Match your model input size

# === Flask app ===
app = Flask(__name__)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)
    return img_array

def postprocess_mask(mask_array):
    mask = (mask_array > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask.squeeze().astype(np.uint8), mode='L')
    return mask_image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    image_bytes = file.read()

    try:
        input_image = preprocess_image(image_bytes)
        prediction = model.predict(input_image)[0]  # shape: (H, W, 1)
        mask_img = postprocess_mask(prediction)

        img_io = io.BytesIO()
        mask_img.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
