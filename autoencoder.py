"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Simple autoencoder for image denoising

    Implemented following instructions on
    https://blog.keras.io/building-autoencoders-in-keras.html
"""

from __future__ import print_function
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.models import model_from_json, Model
from img_to_csv import show_row_image
from add_noise import add_noise

import numpy as np
import os
import sys
import data

# avoids an incompatibility error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(argv):
    """
    trains autoencoder on the lfw dataset and shows its output on a clean picture on my face and on a
    noisy picture from the testset
    """

    if len(argv) >= 2:
        save_filename = argv[1]
    else:
        save_filename = "model"

    y_train = data.Data("lfw_x_train.csv").numericData
    y_test = data.Data("lfw_x_test.csv").numericData

    img_rows = 64
    img_cols = 64

    # the autoencoder output are the original images
    #   convert to ndarrays of dtype float32
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    # the autoencoder input are noisy versions of the original images
    x_train = add_noise(y_train)
    x_test = add_noise(y_test)

    # reshape data according to backend image format
    if K.image_data_format() == 'channels_first':
        y_train = y_train.reshape(y_train.shape[0], 1, img_rows, img_cols)
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

        y_test = y_test.reshape(y_test.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, 1)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

        y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    input_img = Input(shape=input_shape)

    # encoder stack
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    layer = MaxPooling2D((2, 2), padding='same')(layer)
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
    encoded = MaxPooling2D((2, 2), padding='same')(layer)

    # decoder stack
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    layer = UpSampling2D((2, 2))(layer)
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
    layer = UpSampling2D((2, 2))(layer)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, y_train,
                    epochs=130,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_test, y_test))

    my_face = data.Data("lfw_query_mine_x.csv").numericData
    my_face = np.asarray(my_face).astype('float32')

    my_face_predict = autoencoder.predict(np.reshape(my_face, (1, 64, 64, 1)))
    testset_predict = autoencoder.predict(np.reshape(x_test, (len(x_test), 64, 64, 1)))

    # show how the autoencoder prediction for a clean image outsied of testset
    show_row_image(np.reshape(my_face_predict[0], (1, 4096)), 64, 64, "my face")

    # show the noisy and predicted versions of a image in the testset
    show_row_image(np.reshape(x_test[1], (1, 4096)), 64, 64, "input")
    show_row_image(np.reshape(testset_predict[1], (1, 4096)), 64, 64, "predicted")

    # serialize model to JSON
    model_json = autoencoder.to_json()
    with open("auto.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights("auto.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main(sys.argv)





