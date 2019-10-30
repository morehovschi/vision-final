"""
    Marius Orehovschi
    CS 365 S19
    Final Project

    Helper file that reads in images as rows in a csv file, adds noise (black lines)
"""

import cv2
import numpy as np
import sys
import random

import data
import analysis
from img_to_csv import show_row_image

def draw_rect(img, x0, y0, x1, y1):
    """
    draws black rectangle on parameter (numpy ndarray) img bounded by x0, y0, x1, y1
    """

    # clip box coordinates to image coordinates
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(x1, img.shape[1]-1)
    y1 = min(y1, img.shape[0] - 1)

    for i in range(y0, y1):
        for j in range(x0, x1):
            img[i, j] = 0


def add_noise_img(img_row, img_height=64, img_width=64):
    """
    draws between 1 and 5 boxes of randomized height and width on parameter img_row
    :param img_row: a flattened numpy ndarray
    """
    copy = img_row.copy()
    img = np.reshape(copy, (img_height, img_width))

    # repeat i times, where 1 <= i <= 5
    for i in range(random.randint(1, 5)):
        # determine short and long edge
        short = random.randint(1, 2)
        long = random.randint(8, 20)

        # box is vertical only 30% of the time
        if random.random() > 0.3:
            box_height = short
            box_width = long
        else:
            box_height = long
            box_width = short

        # boxes are normally distributed in the image
        x0 = int(random.gauss(img_width/2 - box_width/2, img_width/4))
        y0 = int(random.gauss(img_height/2 - box_height/2, img_height/4))

        draw_rect(img, x0, y0, x0 + box_width, y0+box_height)

    img = np.reshape(img, (img_height * img_width,))

    return img

def add_noise(mat_of_images, img_height=64, img_width=64):
    """
    adds random black lines in each image in parameter mat_of_images
    :param mat_of_images: numpy ndarray containing flattened representations of images with
    parameter height and parameter width
    """

    noised = np.zeros(mat_of_images.shape)
    for i in range(len(mat_of_images)):
        noised[i] = add_noise_img(mat_of_images[i], img_height, img_width)

    return noised

def main(argv):
    """
    test for add_noise
    """

    imgs = data.Data("lfw_x.csv").numericData

    noised = add_noise(imgs)

    for i in range(len(imgs)):
        show_row_image(noised[i], 64, 64)


if __name__ == "__main__":
    main(sys.argv)
