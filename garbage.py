"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Reads in directory of pgm images and writes out a csv data file in which rows correspond to
    a flattened version of each image;
"""

import cv2
import numpy as np
import sys
import os
import random

import data
import pgm
import analysis
import img_to_csv

# def img_to_csv(dirname, img_height=120, img_width=128, omit_sunglasses=True, straight_pose=False):
#     """
#     reads in all pgm images in directory dirname, normalizes them, writes them to file as rows in a data sheet;
#     images must all be of parameter img_height and parameter img_width;
#     (optional) parameter omit_sunglasses determines whether the method will skip over images with subjects
#     wearing sunglasses;
#     (optional) parameter straight pose determines if the program will skip pictures with subjects not looking
#     straight at the camera
#     """
#     cwd = os.getcwd()
#
#     # data object that will store the category of each observation
#     dobj_y = data.Data()
#     dobj_y.headers = {"category": -1}
#     dobj_y.types = ["non numeric"]
#     categories = []
#
#     # make data object that will store images as rows
#     dobj_x = data.Data()
#     headers = {}
#     types = [''] * img_height * img_width
#
#     # rows of data points in x
#     rows = []
#
#     for i in range(img_height):
#         for j in range(img_width):
#             headers["pix(%d;%d)" % (i, j)] = i * img_width + j + 1
#             types[i * img_width + j] = "numeric"
#
#     directory = os.listdir(dirname)
#
#     for subdir_name in directory:
#         try:
#             subdir = os.listdir(dirname + "/" + subdir_name)
#         except NotADirectoryError:
#             continue
#
#         for filename in subdir:
#             # the '2' and '4' tags indicate lower resolution versions of the same images; skip them
#             if filename.find("2") != -1 or filename.find("4") != -1:
#                 continue
#
#             # some images are bad; skip them as well
#             if "bad" in filename:
#                 continue
#
#             if omit_sunglasses and ("sunglasses" in filename):
#                 continue
#
#             if straight_pose and ("straight" not in filename):
#                 continue
#
#             filename = dirname + "/" + subdir_name + "/" + filename
#
#             image = pgm.pgmread(filename)
#
#             img_height, img_width = image[1], image[2]
#
#             image = image[0].astype('float32')
#
#             normalize_container = np.zeros((800, 800))
#             image = cv2.normalize(image, normalize_container, 0, 255, cv2.NORM_MINMAX)
#
#             image /= 255
#
#             # add image as a row in the list of data points
#             rows.append(np.reshape(image, (1, img_height * img_width)))
#
#     dobj_x.headers = headers
#     dobj_x.types = types
#
#     numericData = np.zeros(shape=(len(rows), img_height * img_width))
#     for i in range(len(rows)):
#         numericData[i, :] = rows[i]
#
#     dobj_x.numericData = numericData
#     dobj_x.write(dirname + "_x.csv")



def main(argv):

    mat = np.matrix([[1, 2, 3],
                     [3, 3, 2]])

    print(np.mean(mat, axis=0))

    b=0
    for i in range(200):
        r = random.randint(1, 5)
        if r == 5:
            b+=1

    print(b)


if __name__ == "__main__":
    main(sys.argv)
