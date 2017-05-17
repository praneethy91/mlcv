import numpy as np
import os
from xml.etree import ElementTree
import pickle

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 14
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name == '0_cn':
            one_hot_vector[0] = 1
        elif name == '1_cn':
            one_hot_vector[1] = 1
        elif name == '1_us':
            one_hot_vector[2] = 1
        elif name == '1_eu':
            one_hot_vector[3] = 1
        elif name == '2_cn':
            one_hot_vector[4] = 1
        elif name == '3_cn':
            one_hot_vector[5] = 1
        elif name == '3_us':
            one_hot_vector[6] = 1
        elif name == '3_eu':
            one_hot_vector[7] = 1
        elif name == '4_cn':
            one_hot_vector[8] = 1
        elif name == '5_cn':
            one_hot_vector[9] = 1
        elif name == '6_cn':
            one_hot_vector[10] = 1
        elif name == '7_cn':
            one_hot_vector[11] = 1
        elif name == '8_cn':
            one_hot_vector[12] = 1
        elif name == '9_cn':
            one_hot_vector[13] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

## example on how to use it
# import pickle
data = XML_preprocessor('/Users/xynazog/Downloads/Gieger_MLCV_2017_hand_gestures_pascal_labeled/labels/labels_number_region/').data
pickle.dump(data,open('Hand_First.p','wb'))

