from PIL import Image
from multiprocessing import Pool
import glob
import face_recognition
import multiprocessing
import os
import uuid


# references
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# https://github.com/ageitgey/face_recognition
# https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

TEST = True
IMAGE_SOURCE_ROOT_DIR_TEST = '/srv/www/website_assets/images/2018/aaron_and_alyssa'
IMAGE_SOURCE_ROOT_DIR = '/srv/www/website_assets/images'
RESULT_DIR = '/home/mmorano/face_recognition'

data = []


def save_faces(path, image, face_locations):
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # original code would preserve relative directory structures
        # dest_path = path.replace(image_source_root_dir, result_dir)
        # dest_path = dest_path.replace('md/', str(i) + "_")

        # new code writes to a single directory w/ unique names, to simplify
        # later clustering code
        dest_path = os.path.join(RESULT_DIR, str(uuid.uuid4()) + ".jpg")

        face_dir = os.path.dirname(dest_path)

        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(dest_path)


def find_faces(path):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(path)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    save_faces(path, image, face_locations)

    #if len(face_locations) > 0:
    #    print(path)
    #    d = [{"image_path": path, "loc": loc, "encoding": enc}
    #        for (loc, enc) in zip(face_locations, face_encodings)]
    #    data.extend(d)


def main():
    print('getting list of files...')

    if TEST:
        image_list = glob.glob(IMAGE_SOURCE_ROOT_DIR_TEST + '/md/*.jpg')
    else:
        image_list = glob.glob(IMAGE_SOURCE_ROOT_DIR + '/*/*/md/*.jpg')

    print('found ' + str(len(image_list)) + " images")

    print('processing files in parallel...')
    pool = Pool()
    pool.map(find_faces, image_list)


main()
