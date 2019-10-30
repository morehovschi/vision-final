"""
    Marius Orehovschi
    CS 365 S19

    Simple convolutional neural network based on a Keras template found at
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    trains network on MNIST, plots train and test accuracy of each epoch
    and saves model to disk

    save model code by Jason Brownlee
    https://machinelearningmastery.com/save-load-keras-deep-learning-models/
"""

from __future__ import print_function
import keras
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# avoids an incompatibility error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# global variable for printing the current epoch
completed_epoch = 1

class TestCallback(Callback):
    """
     helper Callback class based on template by joelthchao https://github.com/joelthchao
     stores train and test accuracy after each epoch
    """
    def __init__(self, train_data, test_data, container):
        self.test_data = test_data
        self.train_data = train_data
        self.container = container
    def on_epoch_end(self, epoch, logs={}):
        x_0, y_0 = self.train_data
        x, y = self.test_data
        loss_0, acc_0 = self.model.evaluate(x_0, y_0, verbose=0)
        loss, acc = self.model.evaluate(x, y, verbose=0)

        self.container.append((acc_0, acc))
        # print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        global completed_epoch
        print("done with epoch", completed_epoch)
        completed_epoch += 1


def main(argv):
    """
    saves Keras model serialized to JSON;
    optional name of the model can be specified with argv[1], default is 'model'
    """

    if len(argv) == 2:
        save_filename = argv[1]
    else:
        save_filename = "model"

    batch_size = 128
    num_classes = 10
    epochs = 8

    img_rows = 28
    img_cols = 28

    np.random.seed(42)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape data into a form that can be used for learning
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # convert data type of train and test data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_trainshape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert data type of train and test data
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # make convolution stack
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.375))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # container for accuracies of train and test data
    epoch_accuracies = []

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[TestCallback((x_train, y_train), (x_test, y_test), epoch_accuracies)])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    epoch_accuracies = np.array(epoch_accuracies)

    # plot train and test accuracies over each epoch
    plt.figure(1)
    line1 = plt.plot(range(1, epochs+1), epoch_accuracies[:, 0], label='train accuracy')
    line2, = plt.plot(range(1, epochs+1), epoch_accuracies[:, 1], label='test accuracy')
    plt.xlabel('epoch number')
    plt.legend()
    plt.show()

    # serialize model to JSON
    model_json = model.to_json()
    with open(save_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_filename + ".h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main(sys.argv)





