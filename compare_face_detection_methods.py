import cv2 as cv
import numpy as np
import os as os

# references:
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html

opencv_data_dir = '/home/mmorano/git/opencv/data'

def run_opencv_detector(name, classifier):
    classifier = os.path.join(opencv_data_dir, classifier)
    cv_classifier = cv.CascadeClassifier(classifier)


def main():
    run_opencv_detector('Haar default', 'haarcascades/haarcascade_frontalface_default.xml')
    run_opencv_detector('Haar alt', 'haarcascades/haarcascade_frontalface_alt.xml')
    run_opencv_detector('Haar alt2', 'haarcascades/haarcascade_frontalface_alt2.xml')
    run_opencv_detector('Haar alt tree', 'haarcascades/haarcascade_frontalface_alt_tree.xml')
    run_opencv_detector('Lbp default', 'lbpcascades/lbpcascade_frontalface.xml')
    run_opencv_detector('Lbp improved', 'lbpcascades/lbpcascade_frontalface_improved.xml')

main()
