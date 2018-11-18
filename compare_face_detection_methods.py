import datetime as dt
import os as os
import cv2 as cv
import face_recognition as fr


# references:
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
# https://github.com/ageitgey/face_recognition
opencv_data_dir = '/home/mmorano/git/opencv/data'
photo = '/srv/www/website_assets/images/2018/hotpot/md/MVIMG_20180330_193247.jpg'
refimg = cv.imread(photo)


def print_summary(name, face_count, start_time, end_time):
    print('-----------------------------------------------------------')
    print('detector name: ' + name)
    print('faces found: ' + str(face_count))
    print('time taken: ' + str((end_time - start_time).total_seconds()))
    print()


def show_matches(name, faces):
    demo_img = refimg.copy()

    for (x, y, w, h) in faces:
        cv.rectangle(demo_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow(name, demo_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def run_opencv_detector(name, classifier):
    # prep detector
    name = 'opencv-' + name
    classifier = os.path.join(opencv_data_dir, classifier)
    cv_classifier = cv.CascadeClassifier(classifier)

    # prep image
    img = cv.imread(photo)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detect
    start_time = dt.datetime.now()
    faces = cv_classifier.detectMultiScale(img_gray, )
    end_time = dt.datetime.now()

    # summarize
    print_summary(name, len(faces), start_time, end_time)
    show_matches(name, faces)


def run_face_recognition_detector(model):
    # prep detector
    name = 'face_recognition-' + model

    # prep image
    image = fr.load_image_file(photo)

    # detect
    start_time = dt.datetime.now()
    faces = fr.face_locations(image, number_of_times_to_upsample=1, model=model)
    end_time = dt.datetime.now()
    #face_encodings = face_recognition.face_encodings(image, face_locations)

    # convert faces result to new array based on (x, y, w, h)
    faces_for_show = []
    for (top, right, bottom, left) in faces:
        faces_for_show.append((left, bottom, right - left, top - bottom))

    # summarize
    print_summary(name, len(faces), start_time, end_time)
    show_matches(name, faces_for_show)


def main():
    run_opencv_detector('Haar default',  'haarcascades/haarcascade_frontalface_default.xml')
    run_opencv_detector('Haar alt',      'haarcascades/haarcascade_frontalface_alt.xml')
    run_opencv_detector('Haar alt2',     'haarcascades/haarcascade_frontalface_alt2.xml')
    run_opencv_detector('Haar alt tree', 'haarcascades/haarcascade_frontalface_alt_tree.xml')
    run_opencv_detector('Lbp default',   'lbpcascades/lbpcascade_frontalface.xml')
    run_opencv_detector('Lbp improved',  'lbpcascades/lbpcascade_frontalface_improved.xml')

    run_face_recognition_detector('hog')
    run_face_recognition_detector('cnn')


main()
