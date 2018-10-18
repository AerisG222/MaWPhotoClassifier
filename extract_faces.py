from PIL import Image
from multiprocessing import Pool, Value
from sklearn.cluster import DBSCAN
import glob
import face_recognition
import multiprocessing
import numpy
import os
import uuid


# references
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# https://github.com/ageitgey/face_recognition
# https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

TEST = False
IMAGE_SOURCE_ROOT_DIR_TEST = '/srv/www/website_assets/images/2018/aaron_and_alyssa'
IMAGE_SOURCE_ROOT_DIR = '/srv/www/website_assets/images/2018'
RESULT_DIR = '/home/mmorano/face_recognition'

counter = None
total_count = None


def progressBar(title, value, endvalue, bar_length=40):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        print("{0} Percent: [{1}] {2}%".format(title, arrow + spaces, int(round(percent * 100))), end="\r")


def save_face(src_path, dest_dir, face_location):
    top, right, bottom, left = face_location
    dest_path = os.path.join(dest_dir, str(uuid.uuid4()) + ".jpg")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    image = face_recognition.load_image_file(src_path)

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.save(dest_path)


def find_faces(path):
    global counter
    global total_count

    with counter.get_lock():
        counter.value += 1
        progressBar("Find Faces", counter.value, total_count.value)

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(path)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    #save_faces(path, image, face_locations)

    return [{"image_path": path, "loc": loc, "encoding": enc}
        for (loc, enc) in zip(face_locations, face_encodings)]


def cluster(data):
    encodings = [d["encoding"] for d in data]

    # cluster the embeddings
    print("[INFO] clustering...")
    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)

    # determine the total number of unique faces found in the dataset
    # label of -1 below could suggest outliers, for now, exclude these
    labelIDs = numpy.unique(clt.labels_)
    numUniqueFaces = len(numpy.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    # loop over the unique face integers
    for labelID in labelIDs:
        print("[INFO] faces for face ID: {}".format(labelID))
        face_dir = os.path.join(RESULT_DIR, "face_" + str(labelID))

        idxs = numpy.where(clt.labels_ == labelID)[0]

        for i in idxs:
            save_face(data[i]["image_path"], face_dir, data[i]["loc"])


def main():
    global counter
    global total_count

    print('getting list of files...')

    if TEST:
        image_list = glob.glob(IMAGE_SOURCE_ROOT_DIR_TEST + '/md/*.jpg')
    else:
        image_list = glob.glob(IMAGE_SOURCE_ROOT_DIR + '/*/md/*.jpg')

    counter = Value('i', 0)
    total_count = Value('i', len(image_list))

    print('found {} images'.format(total_count))

    img_list = []

    # build list of arrays, so we can pass more info to the job that runs in
    # parallel to better illustrate where we are in the overall process
    for i, img in enumerate(image_list):
        plist = []
        plist.append(img)
        plist.append(i)
        plist.append(len(image_list))

        img_list.append(plist)

    print('processing files in parallel...')
    pool = Pool()
    data = pool.map(find_faces, image_list)

    # flatten the list
    # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python?page=1&tab=votes#tab-top
    data = [item for sublist in data for item in sublist]

    # remove any instances where faces were not found
    data = [x for x in data if len(x) > 0]

    #print(data)
    print('clustering faces...')
    cluster(data)


main()
