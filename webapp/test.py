import operator

import matplotlib
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from scipy.misc import imread

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def localize(img_path, model, bbox_util, index):
    # path_prefix = './pics/'

    # files = os.listdir(path_prefix)

    inputs = []
    imgs = []

    #img_path = path_prefix + f
    #Change this to in-memory reference

    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    imgs.append(imread(img_path))
    inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))

    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)


    #print(results)
    highestConfidenceStoredROI = False
    for i, img in enumerate(imgs):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_dictionary = {}
        for i, conf in enumerate(det_conf):
            if conf>=0.6:
                top_dictionary[i] = conf
        sorted_top = sorted(top_dictionary.items(), key=operator.itemgetter(1))

        sorted_top = sorted_top[-2:]
        top_indices = []
        for x in sorted_top:
            top_indices.append(x[0])
        #top_indices = []
        #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        top_indices = top_indices[::-1]
        if len(top_indices)==0:
            break
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            xmid = int(round((xmin+xmax)/2))
            ymid = int(round((ymin+ymax)/2))
            xmin = xmid-112
            xmax = xmid+112
            ymin = ymid-112
            ymax = ymid+112

            if xmin<0:
                xmax+=(-xmin)
                xmin=0
            if xmax>640:
                xmin-=(xmax-640)
                xmax=639
            if ymax>360:
                ymin-=(ymax-360)
                ymax = 359
            if ymin<0:
                ymax+=(-ymin)
                ymin=0
            if not highestConfidenceStoredROI:
                global roi
                roi = img[ymin:ymax, xmin:xmax]
                if ymax - ymin == 223 and xmax - xmin == 223:
                    roi = np.pad(roi, ((0, 1), (0, 1), (0, 0)), 'edge')
                print(roi.shape)
                highestConfidenceStoredROI = True
            score = top_conf[i]
            label = int(top_label_indices[i])
    #         label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        s = './output/foo{0}.png'.format(index)

        plt.axis('off')
        plt.savefig(s, bbox_inches='tight')
        plt.close()
    return roi