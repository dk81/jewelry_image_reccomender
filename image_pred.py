import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import tensorflow as tf

from flask import request, jsonify, Flask, render_template, url_for
from flask_cors import CORS

from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Reference: Deeplizard Youtube videos
app = Flask(__name__)
CORS(app) #To make button work

def get_model():
    model = tf.keras.models.load_model('jewelry_classifier_224.h5')
    print("**Keras model is loaded!**")
    return model

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    return image

print(" <<< Keras model being loaded. >>> ")

@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist()
    response = {
            'bracelet': prediction[0][0],
            'earring': prediction[0][1],
            'necklace': prediction[0][2],
            'ring': prediction[0][3]
         }
    # Obtain key with max value as model prediction in response.
    model_predict = max(response, key=response.get)

    # Obtain path for image comparison given model prediction:
    type_path = '/home/learner/flask_sample_app/static/images' + '/' + model_predict
    
    # Store all images in array form in this initialized array:

    images_array = []

    # Read images in the selected path
    # In each image, convert into multidim array and resize.

    for img in os.listdir(type_path):
        img_array = cv2.imread(os.path.join(type_path, img), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (224, 224))
        images_array.append(new_array)
    
    # Initialize Structural Similarity Index Mean (SSIM) scores list:
    ssim_scores = []
    
    # Note user image is processed_image.
    for img_elem in images_array:
        score = ssim(processed_image[0], img_elem, multichannel = True).round(4)
        ssim_scores.append(score)

    # Path of highest SSIM score:

    highest_ssim_path = type_path + '/' + os.listdir(type_path)[np.argmax(ssim_scores)]
    
    
    #return "<img id='img-pred' src='"+highest_ssim_path+"'/>"
    
    # File name in predicted folder name
    html_image = 'static/images/' + model_predict + '/' + os.listdir(type_path)[np.argmax(ssim_scores)]
    
    # Pass html_image path file into show_recc_image.html This .html file ahs the image only.
    return render_template("show_recc_image.html", filename=html_image)

@app.route("/")
def index():
     return render_template('image_pred.html')

'''
@app.route("/img", methods=["POST"])
def rec_image(img_path)
    return render_template("show_recc_image.html", filename = img_path)
'''

if __name__ == "__main__":
    app.run(port=5000)