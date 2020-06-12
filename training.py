import os
import re
import sys
import warnings
import dill
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from skimage.transform import resize



classnum = 7
filename = '*filename for saving*'
row = 256
col = 256


# Data preparation
X_train = np.append(X_normal_train, X_abnormal_train, axis=0)
Y_train = np.append(Y_normal_train, Y_abnormal_train, axis=0)

X_validation = np.append(X_normal_validation, X_abnormal_validation, axis=0)
Y_validation = np.append(Y_normal_validation, Y_abnormal_validation, axis=0)

X_test = np.append(X_normal_test, X_abnormal_test, axis=0)
Y_test = np.append(Y_normal_test, Y_abnormal_test, axis=0)

print("X shape")
print("train: " + str(X_train.shape))
print("validation: " + str(X_validation.shape))
print("Y shape")
print("train: " + str(Y_train.shape))
print("validation: " + str(Y_validation.shape))
print("test: " + str(Y_test.shape))


# Check GPU status
from tensorflow.python.client import device_lib
from keras import backend as K

device_lib.list_local_devices()
K.tensorflow_backend._get_available_gpus()


# One-hot-encoding
Y_train = np_utils.to_categorical(Y_train, classnum)
Y_validation = np_utils.to_categorical(Y_validation, classnum)
Y_test = np_utils.to_categorical(Y_test, classnum)


# Resize image
def resize(X_train, Y_train):    
    X_train_resize = np.zeros((len(X_train),row,col,3))
    
    for idx in range(len(Y_train)):
        img_X0 = X_train[idx,:,:,0]/255.0
        img_X1 = X_train[idx,:,:,1]/255.0
        img_X2 = X_train[idx,:,:,2]/255.0

        img_re_X0 = cv2.resize(img_X0,(row,col),interpolation=cv2.INTER_LINEAR)
        img_re_X1 = cv2.resize(img_X1,(row,col),interpolation=cv2.INTER_LINEAR)
        img_re_X2 = cv2.resize(img_X2,(row,col),interpolation=cv2.INTER_LINEAR)

        X_train_resize[idx,:,:,0] = img_re_X0
        X_train_resize[idx,:,:,1] = img_re_X1
        X_train_resize[idx,:,:,2] = img_re_X2
    
    return X_train_resize

X_train_resize = resize(X_train, Y_train)
X_validation_resize = resize(X_validation, Y_validation)
X_test_resize = resize(X_test, Y_test)

X_train = X_train_resize
X_validation = X_validation_resize
X_test = X_test_resize

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)


# Data augmentation
from keras.preprocessing.image import ImageDataGenerator
import random

seed = 42
batch_size = 16

def data_augment(x, y, X_validation, Y_validation):
    print(x.shape)
    print(y.shape)
    data_gen_args = dict(rotation_range=90,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        # rescale = 1./255, 
                        # featurewise_std_normalization =True,
                        # shear_range=0.2,
                        zoom_range=0.2,                   
                        vertical_flip=True,
                        horizontal_flip=True,
                        fill_mode='nearest')  #use 'constant'??
                        

    X_datagen = ImageDataGenerator(**data_gen_args)
    X_train_augmented = X_datagen.flow(x,y, batch_size=batch_size, shuffle=True,seed=seed)

    X_datagen_val = ImageDataGenerator()
    X_validation_augmented = X_datagen_val.flow(X_validation, Y_validation, batch_size=batch_size, shuffle=True, seed=seed)

    train_generator = X_train_augmented
    validation_generator = X_validation_augmented
    
    return train_generator, validation_generator
    
    
# Training model - Call EfficientNet from https://github.com/qubvel/efficientnet
import efficientnet.keras as efn
from keras import layers, models

def efficientnet():
    base_model = efn.EfficientNetB1(input_shape = (row, col, 3), classes = classnum, weights = 'imagenet', include_top = False)
    
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    predictions = Dense(classnum, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    
    return model
    
    
# Build model
model = efficientnet()


# Multi-GPU
from keras.utils import multi_gpu_model

class ModelMGPU(Model):

    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model
 
    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

model = ModelMGPU(model, gpus=4)


# Optimizer
from keras.optimizers import SGD, Adadelta, Adam, Adamax, Nadam, RMSprop

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) # 1e-3 or 5e-4
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])


# Train time history
import keras
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = [] 

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


# Model fit generator
tb_hist = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=25, min_lr=0.0001, verbose=1)
earlystopper = EarlyStopping(patience=100, verbose=1)

name_weights = str(filename) + ".h5"
checkpointer = ModelCheckpoint(name_weights, verbose=1, save_best_only=True)

train_generator, validation_generator = data_augment(X_train, Y_train, X_validation, Y_validation)

results=model.fit_generator(train_generator,validation_data = validation_generator, 
                            validation_steps = len(X_validation)/batch_size,
                            steps_per_epoch=len(X_train)/batch_size, epochs=100, 
                            callbacks=[reduce_lr, earlystopper, checkpointer, time_callback])

print(model.evaluate(X_test, Y_test))


# Model load
model = load_model(name_weights)
preds_test = model.predict(X_test, verbose=1)


# Evaluation
from sklearn.metrics import f1_score

def evaluation(preds_test):
    X_test_eval = preds_test[:].argmax(axis=1)
    Y_test_eval = Y_test[:].argmax(axis=1)

    score = (X_test_eval == Y_test_eval) * 1

    final_acc = np.sum(score)/len(X_test_eval)
    final_f1 = f1_score(Y_test_eval, X_test_eval, average='macro')
    
    return final_acc, final_f1
    
final_acc, final_f1 = evalutaion(preds_test)
print("Final accuracy: ", final_acc)
print("Final F1 score(average): ", final_f1)


# Metrics
from sklearn.metrics import confusion_matrix

confusion_matrix(X_test_eval, Y_test_eval) # confusion matrix
