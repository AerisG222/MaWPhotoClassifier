import cv2 as cv
import datetime as dt
import numpy as np
import os as os

# references:
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html

opencv_data_dir = '/home/mmorano/git/opencv/data'
photo = '/srv/www/website_assets/images/2018/hotpot/md/MVIMG_20180330_193247.jpg'
img = cv.imread(photo)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def run_opencv_detector(name, classifier):
    classifier = os.path.join(opencv_data_dir, classifier)
    cv_classifier = cv.CascadeClassifier(classifier)
    start_time = dt.datetime.now()
    faces = cv_classifier.detectMultiScale(img_gray, )
    end_time = dt.datetime.now()

    demo_img = img.copy()

    for (x, y, w, h) in faces:
        cv.rectangle(demo_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    print('-----------------------------------------------------------')
    print(name)
    print('faces found: ' + str(len(faces)))
    print('time taken: ' + str((end_time - start_time).total_seconds()))
    print()

    cv.imshow(name, demo_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    run_opencv_detector('Haar default',  'haarcascades/haarcascade_frontalface_default.xml')
    run_opencv_detector('Haar alt',      'haarcascades/haarcascade_frontalface_alt.xml')
    run_opencv_detector('Haar alt2',     'haarcascades/haarcascade_frontalface_alt2.xml')
    run_opencv_detector('Haar alt tree', 'haarcascades/haarcascade_frontalface_alt_tree.xml')
    run_opencv_detector('Lbp default',   'lbpcascades/lbpcascade_frontalface.xml')
    run_opencv_detector('Lbp improved',  'lbpcascades/lbpcascade_frontalface_improved.xml')

main()
