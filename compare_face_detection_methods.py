import datetime
import os
import cv2
import face_recognition
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


def print_summary(name, face_count, start_time, end_time):
    print('-----------------------------------------------------------')
    print('detector name: ' + name)
    print('faces found: ' + str(face_count))
    print('time taken: ' + str((end_time - start_time).total_seconds()))
    print()


def show_matches(name, faces):
    demo_img = refimg.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(demo_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow(name, demo_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_opencv_detector(name, classifier):
    # prep detector
    name = 'opencv-' + name
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
    print_summary(name, len(faces), start_time, end_time)
    show_matches(name, faces)


def run_opencv_dnn_detector():
    # prep detector
    name = 'opencv-dnn'
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

        if(confidence > 0.5):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.push((startX, startY, endX - startX, endY - startY))

    # summarize
    print_summary(name, len(faces), start_time, end_time)
    show_matches(name, faces)


def run_face_recognition_detector(model):
    # prep detector
    name = 'face_recognition-' + model

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
    print_summary(name, len(faces), start_time, end_time)
    show_matches(name, faces_for_show)


def run_mtcnn_detector():
    # prep detector
    name = 'mtcnn'
    detector = MTCNN()

    # prep image
    img = cv2.imread(photo)

    # detect
    start_time = datetime.datetime.now()
    faces = detector.detect_faces(img)
    end_time = datetime.datetime.now()

    # summarize
    print_summary(name, len(faces), start_time, end_time)
    show_matches(name, faces)


def main():
    print("OpenCV version :  {0}".format(cv2.__version__))
    print("face_recognition version: {0}".format(face_recognition.__version__))
    print()

    run_opencv_detector('Haar default',  'haarcascades/haarcascade_frontalface_default.xml')
    run_opencv_detector('Haar alt',      'haarcascades/haarcascade_frontalface_alt.xml')
    run_opencv_detector('Haar alt2',     'haarcascades/haarcascade_frontalface_alt2.xml')
    run_opencv_detector('Haar alt tree', 'haarcascades/haarcascade_frontalface_alt_tree.xml')
    run_opencv_detector('Lbp default',   'lbpcascades/lbpcascade_frontalface.xml')
    run_opencv_detector('Lbp improved',  'lbpcascades/lbpcascade_frontalface_improved.xml')

    # run_opencv_dnn_detector()

    run_face_recognition_detector('hog')
    run_face_recognition_detector('cnn')

    # run_mtcnn_detector()


main()
