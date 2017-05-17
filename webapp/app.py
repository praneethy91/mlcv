#!/bin/python

import os
from flask import Flask, Response, request, abort, render_template, render_template_string, send_from_directory
from PIL import Image
from werkzeug import secure_filename
import io
import tensorflow as tf
import inception_preprocessing

slim = tf.contrib.slim
from PIL import Image
from inception_resnet_v2 import *
import numpy as np

app = Flask(__name__)

WIDTH = 480
HEIGHT = 270


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
def restore_fn(sess):
    return saver.restore(sess, checkpoint_file)

#Get your supervisor
sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)

image_path =''

images = []
images2 = []

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
    images = []

    return render_template(TEMPLATE, **{
        'images': images
    })

@app.route('/', methods = ['POST'])
def upload_file():
    global images
    global images2

    images = []
    images2 = []

    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        global image_path
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
            'src': image_path
        })
        images.append(f.filename)
        images2.append(f.filename)

        return render_template(TEMPLATE, **{
            'images': images,
            'images2': images2
        })

@app.route('/forward', methods = ['POST'])
def classify():
    with sv.managed_session() as sess:

        global image_path
        global images
        global images2

        im = Image.open(image_path).resize((224, 224))
        im = np.array(im)
        im = im.reshape(224, 224, 3)

        print("Image Shape: {0}".format(im.shape))

        predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})

        # print (np.max(predict_values), np.max(logit_values))
        # print (np.argmax(predict_values), np.argmax(logit_values))
        prediction = np.argmax(predict_values)
        prediction_label = prediction_dict[prediction]
        #print("The prediction is: {0}".format(prediction_label))
        classifylabel = str(prediction_label)
        return render_template(TEMPLATE, **{
            'images': images,
            'images2': images2,
            'classifylabel': classifylabel
        })

if __name__ == '__main__':
    app.run(debug=True, host='::', use_reloader=False)
