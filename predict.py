from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import os
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
import cv2

def predict(image_name):

    img = cv2.imread(image_name)
    if(img is None):

        return "image not found"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255-img
    img = np.reshape(img, (1,28,28,1))
    model1 = Sequential()
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    weights = loaded_model.get_weights()
   # model_new.set_weights(weights)
    #model_new.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    res = loaded_model.predict_classes(img, )
    return res
