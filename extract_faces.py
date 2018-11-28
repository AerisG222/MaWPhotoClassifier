import cv2
import datetime
import glob
import multiprocessing as mp
import numpy
import os
import psutil
import uuid


# references
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# https://github.com/ageitgey/face_recognition
# https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

OPENCV_DNN_DIR = '/home/mmorano/git/opencv/samples/dnn/face_detector'
OPENCV_DNN_PROTOTXT = os.path.join(OPENCV_DNN_DIR, 'deploy.prototxt')
OPENCV_DNN_MODEL = os.path.join(OPENCV_DNN_DIR, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
MIN_CONFIDENCE = 0.50

TEST = False
IMAGE_SOURCE_ROOT_DIR_TEST = '/srv/www/website_assets/images/2018/aaron_and_alyssa'
IMAGE_SOURCE_ROOT_DIR = '/srv/www/website_assets/images/2018'
RESULT_DIR = '/home/mmorano/extracted_faces'

face_count = mp.Value('i', 0)
counter = mp.Value('i', 0)
total_count = mp.Value('i', 0)


def progressBar(title, value, endvalue, bar_length=50):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        print("{0} Percent: [{1}] {2}%".format(title, arrow + spaces, int(round(percent * 100))), end="\r")


def increment_face_count(count):
    global face_count

    if count == 0:
        return

    with face_count.get_lock():
        face_count.value += count


def increment_image_count():
    global counter
    global total_count

    with counter.get_lock():
        counter.value += 1
        progressBar("Extract Faces", counter.value, total_count.value)


def print_timing(title, start_time, end_time):
    print("\n")
    print(title + str((end_time - start_time).total_seconds()) + " seconds")


def get_lg_path(file):
    return file.replace("/md/", "/lg/")


def get_image_list():
    print('getting list of files...')

    start_time = datetime.datetime.now()

    if TEST:
        image_list = glob.glob(IMAGE_SOURCE_ROOT_DIR_TEST + '/md/*.jpg')
    else:
        image_list = glob.glob(IMAGE_SOURCE_ROOT_DIR + '/*/md/*.jpg')

    end_time = datetime.datetime.now()

    print_timing('getting list of ' + str(len(image_list)) + ' files took: ', start_time, end_time)

    return image_list


def extract_faces(photo_path):
    increment_image_count()

    # prep detector
    net = cv2.dnn.readNetFromCaffe(OPENCV_DNN_PROTOTXT, OPENCV_DNN_MODEL)

    # prep image
    img = cv2.imread(photo_path)
    lgimg = []
    detected_faces = 0
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # detect
    net.setInput(blob)
    detections = net.forward()

    # scan results to extract faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if(confidence >= MIN_CONFIDENCE):
            if len(lgimg) == 0:
                lgimg = cv2.imread(get_lg_path(photo_path))
                (lh, lw) = lgimg.shape[:2]
                xratio = lw / w
                yratio = lh / h

            box = detections[0, 0, i, 3:7] * numpy.array([w * xratio, h * yratio, w * xratio, h * yratio])
            (startX, startY, endX, endY) = box.astype("int")
            save_face(lgimg, startX, startY, endX, endY)
            detected_faces += 1

    increment_face_count(detected_faces)


def save_face(img, startX, startY, endX, endY):
    face_dest_path = os.path.join(RESULT_DIR, str(uuid.uuid4()) + ".jpg")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    face_image = img[startY:endY, startX:endX]
    cv2.imwrite(face_dest_path, face_image)


def main():
    global total_count

    image_list = get_image_list()

    total_count.value = len(image_list)

    print('extracting faces in parallel...')

    start_time = datetime.datetime.now()

    threads = psutil.cpu_count(logical = True) - 1

    if threads < 1:
        threads = 1

    pool = mp.Pool(processes = threads)
    pool.map(extract_faces, image_list)

    end_time = datetime.datetime.now()

    print_timing('extracted ' + str(face_count.value) + ' faces across ' + str(len(image_list)) + ' images took: ', start_time, end_time)


main()
