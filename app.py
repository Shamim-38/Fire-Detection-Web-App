import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
#import tensorflow
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil



app = Flask(__name__)



print('Model loaded. Check http://127.0.0.1:5000/')


MODEL_PATH = 'models/model.h5'


model = load_model(MODEL_PATH)



def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)



        # Make prediction
        preds = model_predict(img, model)
        print(preds)
        print(preds[0][0])
        print(preds[0][1])
        
        if preds[0][0] >= .50:
            result = "Fire"
        else:
            result = "Not FIre"
          
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
    
        result = result.replace('_', ' ').capitalize()
        
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result,  probability=pred_proba)

    return None


if __name__ == '__main__':

    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
