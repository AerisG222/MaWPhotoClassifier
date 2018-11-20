import datetime
import numpy
import os
import cv2
import face_recognition
import texttable
#from mtcnn.mtcnn import MTCNN


# references:
# https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
# https://github.com/ageitgey/face_recognition
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/#
# https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
# https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/
opencv_data_dir = '/home/mmorano/git/opencv/data'
opencv_dnn_dir = '/home/mmorano/git/opencv/samples/dnn/face_detector'
photo = '/srv/www/website_assets/images/2018/hotpot/md/MVIMG_20180330_193247.jpg'
refimg = cv2.imread(photo)
result_summaries = [("Detector", "Faces Found", "Duration (s)")]
result_images = []
result_images_per_row = 4


def show_results():
    print_summary()
    show_comparison()


def print_summary():
    t = texttable.Texttable()
    t.add_rows(result_summaries)
    print(t.draw())


def print_status(name):
    print('processing ' + name + '...')


def scale_image(img):
    return cv2.resize(img, None, fx=0.4, fy=0.4)


def show_comparison():
    for idx, img in enumerate(result_images):
        result_images[idx] = scale_image(img)

    # break array of images into array of arrays of images [to produce grid of result images]
    rows = [result_images[i : i + result_images_per_row] for i in range(0, len(result_images), result_images_per_row)]

    # fill the last row with black images if needed (required by vstack)
    if len(rows) > 1:
        first_img = rows[0][0]
        len_first_row = len(rows[0])
        blank = numpy.zeros(first_img.shape, numpy.uint8)
        blank[:] = ( 32, 32, 32)

        while len(rows[-1]) < len_first_row:
            rows[-1].append(blank)

    result_rows = []

    # horizontally stack multiple images for each row, yielding one image per row
    for row in rows:
        result_rows.append(numpy.hstack(row))

    # vertically stack the rows to finish the image grid
    result_img = numpy.vstack(result_rows)

    cv2.imshow('Results', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_summary(name, face_count, start_time, end_time):
    result_summaries.append( (name, face_count, (end_time - start_time).total_seconds()) )


def add_matches_visual(name, faces):
    demo_img = refimg.copy()

    cv2.putText(demo_img, name, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in faces:
        cv2.rectangle(demo_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    result_images.append(demo_img)


def run_opencv_detector(name, classifier):
    name = 'opencv-' + name
    print_status(name)

    # prep detector
    classifier = os.path.join(opencv_data_dir, classifier)
    cv_classifier = cv2.CascadeClassifier(classifier)

    # prep image
    img = cv2.imread(photo)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect
    start_time = datetime.datetime.now()
    faces = cv_classifier.detectMultiScale(img_gray, )
    end_time = datetime.datetime.now()

    # summarize
    add_summary(name, len(faces), start_time, end_time)
    add_matches_visual(name, faces)


def run_opencv_dnn_detector(min_confidence):
    name = 'opencv-dnn- ' + str(min_confidence * 100) + '% confidence'
    print_status(name)

    # prep detector
    prototxt = os.path.join(opencv_dnn_dir, 'deploy.prototxt')
    model = os.path.join(opencv_dnn_dir, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # prep image
    img = cv2.imread(photo)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # detect
    start_time = datetime.datetime.now()
    net.setInput(blob)
    detections = net.forward()
    end_time = datetime.datetime.now()

    # scan results to extract faces
    faces_for_show = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if(confidence >= min_confidence):
            box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces_for_show.append((startX, startY, endX - startX, endY - startY))

    # summarize
    add_summary(name, len(faces_for_show), start_time, end_time)
    add_matches_visual(name, faces_for_show)


def run_face_recognition_detector(model):
    name = 'face_recognition-' + model
    print_status(name)

    # prep detector

    # prep image
    image = face_recognition.load_image_file(photo)

    # detect
    start_time = datetime.datetime.now()
    faces = face_recognition.face_locations(image, number_of_times_to_upsample=1, model=model)
    end_time = datetime.datetime.now()
    #face_encodings = face_recognition.face_encodings(image, face_locations)

    # convert faces result to new array based on (x, y, w, h)
    faces_for_show = []
    for (top, right, bottom, left) in faces:
        faces_for_show.append((left, bottom, right - left, top - bottom))

    # summarize
    add_summary(name, len(faces), start_time, end_time)
    add_matches_visual(name, faces_for_show)


def run_mtcnn_detector():
    name = 'mtcnn'
    print_status(name)

    # prep detector
    detector = MTCNN()
    print_status(name)

    # prep image
    img = cv2.imread(photo)

    # detect
    start_time = datetime.datetime.now()
    faces = detector.detect_faces(img)
    end_time = datetime.datetime.now()

    # summarize
    add_summary(name, len(faces), start_time, end_time)
    add_matches_visual(name, faces)


def main():
    print("OpenCV version :  {0}".format(cv2.__version__))
    print("face_recognition version: {0}".format(face_recognition.__version__))
    #print("mtcnn version: {0}".format(mtcnn.__version__))
    print()

    run_opencv_detector('Haar default',  'haarcascades/haarcascade_frontalface_default.xml')
    run_opencv_detector('Haar alt',      'haarcascades/haarcascade_frontalface_alt.xml')
    run_opencv_detector('Haar alt2',     'haarcascades/haarcascade_frontalface_alt2.xml')
    run_opencv_detector('Haar alt tree', 'haarcascades/haarcascade_frontalface_alt_tree.xml')
    run_opencv_detector('Lbp default',   'lbpcascades/lbpcascade_frontalface.xml')
    run_opencv_detector('Lbp improved',  'lbpcascades/lbpcascade_frontalface_improved.xml')

    run_opencv_dnn_detector(0.10)
    run_opencv_dnn_detector(0.30)
    run_opencv_dnn_detector(0.50)
    run_opencv_dnn_detector(0.70)
    run_opencv_dnn_detector(0.90)

    run_face_recognition_detector('hog')
    run_face_recognition_detector('cnn')

    # run_mtcnn_detector()

    show_results()


main()
