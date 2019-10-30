"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Reads in csv data file, performs Principal Component Analysis, writes out a PCA file with the minimum number of
    eigenvectors necessary to span >=95% of the variance in the data
"""

import cv2
import numpy as np
import sys

import data
import analysis
from img_to_csv import show_row_image


def main(argv):
    """
    To run eigenface approximation:
        - make sure that the image database and the query image directory have been processed with img_to_csv.py
        - specify the image database ('cmu' or 'lfw') and the query image directory ('cmu_query', 'lfw_query', or
        'lfw_query_mine')
        - specify height and width of images in the dataset ('120, 128' for cmu; '64, 64' for lfw)
    """

    if len(argv) < 5:
        print("Usage: %s <original images directory> <query image directory> <image height> <image width>" % argv[0])
        exit(0)
    else:
        # store names of image directories and dimensions of the images in them
        original_images = argv[1]
        query_image = argv[2]
        img_height = int(argv[3])
        img_width = int(argv[4])

    # find the average of the images in directory
    face_data = data.Data(original_images + "_x.csv")
    avg_img = np.mean(face_data.numericData, axis=0)
    show_row_image(avg_img, img_height, img_width, "average image")

    # read in query image from its directory
    query = data.Data(query_image + "_x.csv").numericData[0]
    show_row_image(query, img_height, img_width, "unprocessed")

    # make a centered version of the original image data
    centered = query - avg_img
    show_row_image(centered, img_height, img_width, "mean-subtracted")

    # read in eigenface data file
    evec_data = data.Data(original_images + "_evec.csv")
    evecs = evec_data.numericData

    # coefficients that determine how much each principal component adds to an image approximation
    coefficients = []

    for i in range(evec_data.get_num_dimensions()):
        coefficients.append(np.dot(centered, evecs[:, i])[0, 0])

    # container for approximated image
    corrected = np.zeros(centered.shape)

    # approximated image is made up of a linear combination of principal components, each scaled by the dot product
    #   of itself with the original image
    for i in range(evec_data.get_num_dimensions()):
        corrected += coefficients[i] * evecs[:, i].T

    corrected += avg_img
    show_row_image(corrected, img_height, img_width, "corrected")

if __name__ == "__main__":
    main(sys.argv)
