import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def UNet(input_size = (512, 512, 1), pretrained_weights = None):

    Inputs = Input(input_size)

    Conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Inputs)
    Conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv1)
    Pool1 = MaxPooling2D(pool_size = (2, 2))(Conv1)

    Conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Pool1)
    Conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2)
    Pool2 = MaxPooling2D(pool_size = (2, 2))(Conv2)

    Conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Pool2)
    Conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv3)
    Pool3 = MaxPooling2D(pool_size = (2, 2))(Conv3)

    Conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Pool3)
    Conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv4)
    Drop4 = Dropout(0.5)(Conv4)
    Pool4 = MaxPooling2D(pool_size = (2, 2))(Drop4)

    Conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Pool4)
    Conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv5)
    Drop5 = Dropout(0.5)(Conv5)

    Up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Drop5))
    Merge6 = concatenate([Drop4,Up6], axis = 3)
    Conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge6)
    Conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv6)

    Up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Conv6))
    Merge7 = concatenate([Conv3,Up7], axis = 3)
    Conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge7)
    Conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv7)

    Up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Conv7))
    Merge8 = concatenate([Conv2,Up8], axis = 3)
    Conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge8)
    Conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv8)

    Up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Conv8))
    Merge9 = concatenate([Conv1,Up9], axis = 3)
    Conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
    Conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv9)
    Conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv9)

    Conv10 = Conv2D(1, 1, activation = 'sigmoid')(Conv9)

    model = Model(inputs = Inputs, output = Conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
      model.load_weights(pretrained_weights)

    return model

UNet()