#!/bin/python

import io

import tensorflow as tf
from flask import Flask, Response, request, abort, render_template, send_from_directory
from werkzeug import secure_filename
import time

import inception_preprocessing

slim = tf.contrib.slim
from PIL import Image
from inception_resnet_v2 import *
from test import localize

from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import os
from ssd import SSD300
from ssd_utils import BBoxUtility

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

app = Flask(__name__)

WIDTH = 480
HEIGHT = 270

counter = 0

TEMPLATE = 'index.html'
UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#------------------------------------------------------------------------

prediction_dict = {0: '0', 1: '1', 2: '2', 3: '4', 4: '5', 5: 'CN_3', 6: 'CN_6',
                   7: 'CN_7', 8: 'CN_8', 9: 'CN_9', 10: 'UK_3', 11: 'US_3'}

# State your log directory where you can retrieve your model
log_dir = '../log'
checkpoint_file = tf.train.latest_checkpoint(log_dir)

#Create a new evaluation log directory to visualize the validation process
log_eval = '../log_eval_test'

input_tensor = tf.placeholder(tf.float32, shape=(224,224,3), name='input_image')
scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
scaled_input_tensor = inception_preprocessing.preprocess_image(scaled_input_tensor, 224, 224, is_training=False)
scaled_input_tensor = tf.reshape(scaled_input_tensor, (1, 224, 224, 3))

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(scaled_input_tensor, num_classes=12, is_training=False)

variables_to_restore = slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
sess = tf.Session()
saver.restore(sess, checkpoint_file)

image_path =''
image2_path=''

images = []
images2 = []
classifylabel = ''

#------------------------------------------------------------------------

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

NUM_CLASSES = 2
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights.29-1.02.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

#------------------------------------------------------------------------

@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        ios = io.StringIO()
        im.save(ios, format='PNG')
        return Response(ios.getvalue(), mimetype='image/png')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)

@app.route('/')
def index():
    global images
    global images2
    global classifylabel

    return render_template(TEMPLATE, **{
        'images': images,
        'images2': images2,
        'classifylabel': classifylabel
    })

@app.route('/', methods = ['POST'])
def upload_file():
    global images
    global images2
    global image_path
    global image2_path

    images = []
    images2 = []

    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image2_path = image_path
        f.save(image_path)
        im = Image.open(image_path)
        w, h = im.size

        aspect = 1.0*w/h

        if aspect > 1.0*WIDTH/HEIGHT:
            width = min(w, WIDTH)
            height = width/aspect
        else:
            height = min(h, HEIGHT)
            width = height*aspect

        images.append({
            'width': int(width),
            'height': int(height),
            'src': image_path
        })

        images2.append({
            'width': int(width),
            'height': int(height),
            'src': image2_path
        })
        images.append(f.filename)
        images2.append(f.filename)

        return render_template(TEMPLATE, **{
            'images': images,
            'images2': images2
        })

@app.route('/forward', methods = ['POST'])
def classify():
    global image_path
    global image2_path
    global images
    global images2
    global classifylabel
    global counter

    images2 = []

    counter = int(time.time())
    im = localize(image_path, model, bbox_util, counter)

    im = im.reshape(224, 224, 3)

    print("Image Shape: {0}".format(im.shape))

    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})

    # print (np.max(predict_values), np.max(logit_values))
    # print (np.argmax(predict_values), np.argmax(logit_values))
    prediction = np.argmax(predict_values)
    prediction_label = prediction_dict[prediction]
    # print("The prediction is: {0}".format(prediction_label))
    classifylabel = str(prediction_label)

    im = Image.open('output/foo{0}.png'.format(counter))
    w, h = im.size

    aspect = 1.0 * w / h

    if aspect > 1.0 * WIDTH / HEIGHT:
        width = min(w, WIDTH)
        height = width / aspect
    else:
        height = min(h, HEIGHT)
        width = height * aspect

    images2.append({
                'width': int(width),
                'height': int(height),
                'src': 'output/foo{0}.png'.format(counter)
            })

    return render_template(TEMPLATE, **{
        'images': images,
        'images2': images2,
        'classifylabel': classifylabel
    })

if __name__ == '__main__':
    app.run(debug=True, host='::', use_reloader=False)
