"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Test file for trained autoencoder
"""

from keras import backend as K
from keras.models import model_from_json
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
    loads test set that was used for training and an image of my face that was not used for training;
    displays the noisy versions of teh images and the ones restored with autoencoding
    """

    y_test = data.Data("lfw_x_test.csv").numericData

    img_rows = 64
    img_cols = 64

    # the autoencoder output are the original images
    #   convert to ndarray of dtype float32
    y_test = np.asarray(y_test).astype('float32')

    # the autoencoder inputs are noisy versions of the original images
    x_test = add_noise(y_test)

    # reshape data according to backend image format
    if K.image_data_format() == 'channels_first':
        y_test = y_test.reshape(y_test.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # load json and create model
    json_file = open('auto.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder.load_weights("auto.h5")
    print("Loaded model from disk")

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # load an image of my face and add noise to it
    my_face = data.Data("lfw_query_mine_x.csv").numericData
    my_face = np.asarray(my_face).astype('float32')
    my_face = add_noise(my_face)

    # load deteriorated image of Bill Gates
    bill = data.Data("lfw_query_x.csv").numericData
    bill = np.asarray(bill).astype('float32')

    # feed the noisy image of my face, the noisy test data, and the deteriorated Bill photo to the autoencoder
    my_face_predict = autoencoder.predict(np.reshape(my_face, (1, 64, 64, 1)))
    testset_predict = autoencoder.predict(np.reshape(x_test, (len(x_test), 64, 64, 1)))
    bill_predict = autoencoder.predict(np.reshape(bill, (1, 64, 64, 1)))

    # show how the noisy image of my face and the autoencoder prediction for it
    show_row_image(my_face[0], 64, 64, "my face with noise")
    show_row_image(np.reshape(my_face_predict[0], (1, 4096)), 64, 64, "my face denoised")

    # show the deteriorated Bill Gates photo and the autoencoder prediction for it
    show_row_image(bill[0], 64, 64, "deteriorated Bill")
    show_row_image(bill_predict[0], 64, 64, "denoised Bill")

    # show the noisy and predicted versions of an image in the testset
    show_row_image(np.reshape(x_test[7], (1, 4096)), 64, 64, "deteriorated testset image")
    show_row_image(np.reshape(testset_predict[7], (1, 4096)), 64, 64, "denoised testset image")


if __name__ == "__main__":
    main(sys.argv)





