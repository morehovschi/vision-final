"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Reads in csv data file, performs Principal Component Analysis, writes out a PCA file with the minimum number of
    eigenvectors necessary to span >=95% of the variance in the data

    Assumes csv data file was written with img_to_csv.py
"""

import cv2
import numpy as np
import sys

import data
import analysis
from img_to_csv import show_row_image


def main(argv):
    """
    performs PCA on the directory given as the first command line argument and writes the results to a file;
    to visualize the average image and the first 6 eigenfaces in the dataset, simply specify the height and the
    width of the images in the target dataset;
    NOTE: the height and width specification is not flexible, the correct height and width are needed for the program
    to be able to display the images
    """

    # default image directory is 'faces'
    if len(argv) < 2:
        dirname = "cmu"
    else:
        dirname = argv[1]

    # if user specified height and width
    if len(argv) == 4:
        img_height = int(argv[2])
        img_width = int(argv[3])

    # read in image data object; follows naming convention in img_to_csv.py
    img_data = data.Data(dirname+'_x.csv').numericData

    centered = img_data - np.mean(img_data, axis=0)

    # if user specified height and width
    if len(argv) == 4:
        show_row_image(np.mean(img_data, axis=0), img_height, img_width, window_label="average image")

    # perform singular value decomposition
    U, S, VT = np.linalg.svd(centered, full_matrices=False)

    # if user specified height and width
    if len(argv) == 4:
        # show normalized versions of the first 6 eigenfaces
        for i in range(6):
            row = VT[i]
            normalized = cv2.normalize(row, np.zeros((img_height, img_width)), 0.0, 1.0, cv2.NORM_MINMAX)
            show_row_image(normalized, img_height, img_width, window_label="eigenface %d" % (i+1))

    # find eigenvalues and cumulative eigenvalues
    evalues = S * S / (img_data.shape[0] - 1)
    cu_eig = np.array(analysis.getCumulativeValueList(evalues))

    # find the index of the last principal component needed to span >=99% of the variance
    min_useful_idx = np.where(cu_eig >= 0.99)[0][0]

    # only keep enough eigenvectors to span 99% of the variance in the data
    evecs = VT[:min_useful_idx+1].T

    # make and write out eigenvector data object
    headers = {}
    types = []

    for i in range(min_useful_idx+1):
        headers["pca%d" % i] = i+1
        types.append("numeric")

    evec_data = data.Data()
    evec_data.headers = headers
    evec_data.types = types
    evec_data.numericData = evecs

    evec_data.write(dirname+"_evec.csv")


if __name__ == "__main__":
    main(sys.argv)
