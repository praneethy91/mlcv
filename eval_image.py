import tensorflow as tf
from train_flowers import get_split
import inception_preprocessing

slim = tf.contrib.slim
from PIL import Image
from inception_resnet_v2 import *
import numpy as np

prediction_dict = {0: '0', 1: '1', 2: '2', 3: '4', 4: '5', 5: 'CN_3', 6: 'CN_6',
                   7: 'CN_7', 8: 'CN_8', 9: 'CN_9', 10: 'UK_3', 11: 'US_3'}

# State your log directory where you can retrieve your model
log_dir = './log'
checkpoint_file = tf.train.latest_checkpoint(log_dir)

# Image to classify
image = 'hands/val/1/bright_ch01_5fps_640x360_000220.png'

#Create a new evaluation log directory to visualize the validation process
log_eval = './log_eval_test'

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

#Now we are ready to run in one session
with sv.managed_session() as sess:

    im = Image.open(image).resize((224,224))
    im = np.array(im)
    im = im.reshape(224,224,3)

    print("Image Shape: {0}".format(im.shape))

    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})

    # print (np.max(predict_values), np.max(logit_values))
    # print (np.argmax(predict_values), np.argmax(logit_values))
    prediction = np.argmax(predict_values)
    prediction_label = prediction_dict[prediction]
    print("The prediction is: {0}".format(prediction_label))