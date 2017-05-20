# Hand Gestures - Classification with Object Detection
This project demonstrates the use of transfer learning using an ensemble of three residual nets and one Inception-v4 of neural networks to classify a set of hand gestures showing the numbers - 1 to 10, in various languages. The architecture used is demonstrated in this paper - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261).

Along with classification we also use a single object detection deep neural net whose architecture is described in this paper - [SSD: Single Shot MultiBox detector](https://arxiv.org/abs/1512.02325). It is capable of detection different classes of objects, even though they occur multiple times each in each image.

We utilize Tensorflow for both the object detection and classification tasks. The training data consisted of a set of images showcasing hand gestures for number signs with the bounding box for the hands(some images have one hands and the other images have two hands) in PASCAL VOC format for object detection. For classification the images were also annotated in these categories:
1. Zero
2. One
3. Two
4. Chinese three
5. US three
6. UK three
7. Four
8. Five
9. Chinese six
10. Chinese seven
11. Chinese eight
12. Chinese nine

We utilize Python with a Flask based backend for the web application which loads these neural net models into RAM and presents a user interface for classifying and localizing a hand signal image.

The interface of the web app and the results appear as below:

![Classification and Localization example](https://github.com/praneethy91/mlcv/blob/master/classification_example.png "Classification and Localization example")
