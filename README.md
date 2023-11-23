# Computer Vision Final Project
Final project in CS 365: Computer Vision taught by Bruce Maxwell at Colby College in the spring of 2019.

An exploration of random patch noise removal from grayscale images with linear (Principal Component Analysis) and non-linear (Autoencoder) techniques.

### DEMO:
The demo file, evaluate_autoencoder.py, loads the trained autoencoder model and demonstrates its performance in restoring both manually and automatically deteriorated images of faces. 

The run requires packages keras, tensorflow 1.15, and opencv. To install the required packages, simply run the shell script install_necessary_packages or install the packages manually.

### RUN:
python3 evaluate_autoencoder.py

![Artifact removal](https://github.com/morehovschi/vision-final/blob/master/screenshot.png?raw=true)
