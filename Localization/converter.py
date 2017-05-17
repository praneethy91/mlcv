import json
import pickle
import numpy as np

#Loading the annotations file
annotations_data = json.load(open("annotations.json", "r"))["result"]

#Creating an annotations dictionary
annotations = {}

#Predefining our conditions of width and height
WIDTH = 640.0
HEIGHT = 360.0

for f in annotations_data:
    if len(f["objects"]) == 0:
        continue
    objs = []
    for obj in f["objects"]:
        objs.append([obj["xmin"] / WIDTH, obj["ymin"] / HEIGHT, obj["xmax"] / WIDTH, obj["ymax"] / HEIGHT, 1])
    annotations[f["file"] + ".png"] = np.array(objs)

#Dumping the dictionary data
with open("hand_first.pkl", "wb") as outf:
    pickle.dump(annotations, outf)