"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Reads in directory of pgm images and writes out a csv data file in which rows correspond to
    a flattened version of each image;

    Uses two similar methods, each designed for a specific dataset
"""

import cv2
import numpy as np
import scipy
import sys
import os

import data
import pgm


def img_to_csv(dirname, omit_sunglasses=True, straight_pose=False):
    """
    reads in all pgm images in directory dirname, writes them to file as rows in a data sheet;
    (optional) parameter omit_sunglasses determines whether the method will skip over images with subjects
    wearing sunglasses;
    (optional) parameter straight pose determines if the program will skip pictures with subjects not looking
    straight at the camera

    written for the CMU faces dataset

    :return: image height, image width
    """

    # rows of image data points
    rows = []

    directory = os.listdir(dirname)

    img_height = None
    img_width = None

    for subdir_name in directory:
        try:
            subdir = os.listdir(dirname + "/" + subdir_name)
        except NotADirectoryError:
            continue

        for filename in subdir:
            # the '2' and '4' tags indicate lower resolution versions of the same images; skip them
            if filename.find("2") != -1 or filename.find("4") != -1:
                continue

            # some images are bad; skip them as well
            if "bad" in filename:
                continue

            # omit portraits with sunglasses and ones where the subject's pose is not straight
            if omit_sunglasses and ("sunglasses" in filename):
                continue
            if straight_pose and ("straight" not in filename):
                continue

            filename = dirname + "/" + subdir_name + "/" + filename

            image = pgm.pgmread(filename)

            img_width, img_height = image[1], image[2]

            # convert to float, normalize, and scale
            image = image[0].astype('float32')
            image /= 255

            # add image as a row in the list of data points
            rows.append(np.reshape(image, (1, img_height * img_width)))

    # make data object that will store images as rows
    dobj_x = data.Data()
    headers = {}
    types = [''] * img_height * img_width

    # name columns as coordinates of corresponding pixels
    for i in range(img_height):
        for j in range(img_width):
            headers["pix(%d;%d)" % (i, j)] = i * img_width + j + 1
            types[i * img_width + j] = "numeric"

    dobj_x.headers = headers
    dobj_x.types = types

    # create a numeric data container for dobj_x and assign the image data rows to it
    numericData = np.zeros(shape=(len(rows), img_height * img_width))
    for i in range(len(rows)):
        numericData[i, :] = rows[i]

    dobj_x.numericData = numericData
    dobj_x.write(dirname + "_x.csv")

    return img_height, img_width

def new_img_to_csv(dirname):
    """
        reads in all pgm images in directory dirname, writes them to file as rows in a data sheet;

        written for the lfw dataset

        :return: image height, image width
        """

    # rows of image data points
    rows = []

    directory = os.listdir(dirname)

    img_height = None
    img_width = None

    for filename in directory:

        if ".pgm" not in filename:
            continue

        filename = dirname + "/" + filename

        image = pgm.read_pgm(filename)

        img_width, img_height = image.shape[1], image.shape[0]

        # convert to float, normalize, and scale
        image = image.astype('float32')
        image /= 255

        # add image as a row in the list of data points
        rows.append(np.reshape(image, (1, img_height * img_width)))

    # make data object that will store images as rows
    dobj_x = data.Data()
    headers = {}
    types = [''] * img_height * img_width

    # name columns as coordinates of corresponding pixels
    for i in range(img_height):
        for j in range(img_width):
            headers["pix(%d;%d)" % (i, j)] = i * img_width + j + 1
            types[i * img_width + j] = "numeric"

    dobj_x.headers = headers
    dobj_x.types = types

    # create a numeric data container for dobj_x and assign the image data rows to it
    numericData = np.zeros(shape=(len(rows), img_height * img_width))
    for i in range(len(rows)):
        numericData[i, :] = rows[i]

    dobj_x.numericData = numericData
    dobj_x.write(dirname + "_x.csv")

    return img_height, img_width


def show_row_image(row, img_height, img_width, window_label=None):
    """
    resizes parameter row into an image with dimensions img_height and img_width and shows image
    """

    image = np.reshape(row, (img_height, img_width))

    if window_label is None:
        window_label = "row_image"

    cv2.imshow(window_label, image)
    cv2.waitKey()


def main(argv):
    """
    opens directory of images, reads images as rows of pixel values, and writes them to file
    """

    if len(argv) < 2:
        dirname = 'lfw'
    else:
        dirname = argv[1]

    if dirname == "cmu" or dirname == "lfw_query" or dirname == "cmu_query" or dirname == "lfw_query_mine_det":
        img_height, img_width = img_to_csv(dirname, straight_pose=True)
    elif dirname == "lfw" or dirname == "lfw_query_mine":
        img_height, img_width = new_img_to_csv(dirname)

    if len(argv) == 3:
        if argv[2] == "show":
            img_data = data.Data(dirname + "_x.csv")

            for i in range(img_data.get_num_points()):
                show_row_image(img_data.numericData[i], img_height, img_width)


if __name__ == "__main__":
    main(sys.argv)
