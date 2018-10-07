from PIL import Image
import glob
import face_recognition
import multiprocessing
import os
import uuid


image_source_root_dir = '/srv/www/website_assets/images'
result_dir = '/home/mmorano/face_recognition'
processes = []
image_list = glob.glob(image_source_root_dir + '/*/*/md/*.jpg')


def save_faces(path, image, face_locations):
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location

        # original code would preserve relative directory structures
        # dest_path = path.replace(image_source_root_dir, result_dir)
        # dest_path = dest_path.replace('md/', str(i) + "_")

        # new code writes to a single directory w/ unique names, to simplify
        # later clustering code
        dest_path = os.path.join(result_dir, str(uuid.uuid4()) + ".jpg")

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

    save_faces(path, image, face_locations)


for img in image_list:
    p = multiprocessing.Process(target = find_faces, args = (img, ))
    processes.append(p)
    p.start()

for proc in processes:
    proc.join()
