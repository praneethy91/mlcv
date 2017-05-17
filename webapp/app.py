#!/bin/python

import os
from flask import Flask, Response, request, abort, render_template, render_template_string, send_from_directory
from PIL import Image
from werkzeug import secure_filename
import io

app = Flask(__name__)

WIDTH = 480
HEIGHT = 270


TEMPLATE = 'index.html'
UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    images = []
    images2 = []

    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        im = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
            'src': os.path.join(app.config['UPLOAD_FOLDER'], filename)
        })

        images2.append({
            'width': int(width),
            'height': int(height),
            'src': os.path.join(app.config['UPLOAD_FOLDER'], filename)
        })
        images.append(f.filename)
        images2.append(f.filename)

        return render_template(TEMPLATE, **{
            'images': images,
            'images2': images2
        })

def localize():
    images2 = []

if __name__ == '__main__':
    app.run(debug=True, host='::')
