# mikeandwan.us Photo Classifier

A utility to try and automatically identify people in photos.  Details below are relatively unorganized notes, but hopefully will be cleaned up if I can get this working.

## Install TensorFlow Docker Container

- https://www.tensorflow.org/install/
- make sure docker is running
    - `sudo systemctl start docker.service`
- `docker pull tensorflow/tensorflow`

## Install Tensorflow Python library

- we install both python2 and python3 libraries as currently tensorflow is not supported on python3.7 [which is what is on Fedora 29, so we will try python2 for now]

```
sudo dnf install boost-devel \
                 cmake \
                 gcc-c++ \
                 graphviz \
                 lapack-devel \
                 openblas-devel \
                 pylint \
                 python2-devel \
                 python3-devel \
                 python2-dlib \
                 python3-dlib \
                 python2-h5py \
                 python3-h5py \
                 python2-matplotlib \
                 python3-matplotlib \
                 python2-opencv \
                 python3-opencv \
                 python2-pydot \
                 python3-pydot \
                 python2-scipy \
                 python3-scipy \
                 python2-tkinter \
                 python3-tkinter
```

- `pip install tensorflow --user`

## References

- Pluralsight - Tensorflow Getting Started by Jerry Kurata

## Install CUDA

- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Fedora&target_version=27&target_type=rpmnetwork
- `sudo dnf install cuda-repo-fedora27-10.0.130-1.x86_64.rpm`
- `sudo dnf clean all`
- `sudo dnf install cuda`

- add cuda to PATH and LD_LIBRARY_PATH

## Install cuDNN

- https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

## Install face_recognition

- `pip install face_recognition --user`
- `pip3 install face_recognition --user`

## Install mtcnn

- oh no, mtcnn requires python3 so the first does not work =(
- `pip install mtcnn --user`
- `pip3 install mtcnn --user`

## Install scikit-learn

- `pip3 install scikit-learn --user`

## Install psutil

- `pip3 install psutil --user`

## Install nvidia-docker2

- https://github.com/NVIDIA/nvidia-docker
- docker build .
- docker run --runtime-nvidia --rm ???

## Build latest opencv

  - install dependencies
    - `sudo dnf install gtk2-devel libdc1394-devel libv4l-devel ffmpeg-devel gstreamer-plugins-base-devel libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel libtiff-devel libwebp-devel tbb-devel eigen3-devel`
  - clone git repo
  - cd opencv
  - mkdir build
  - cd build
  - cmake ..
  - make
  - sudo make install
  - vi ~/.bashrc
    - export PYTHONPATH=/usr/local/python/cv2/python-3.7/:$PYTHONPATH


## Approach

There are a few discrete steps that will be used in the current implementation:

1. Test various face detection techniques for accuracy and performance, and decide on one approach to use for processing image dataset  (compare_face_detection_methods.py)
2. Run script to detect and extract faces and metadata from source images
3. Run script to cluster faces into directories to try and automate labelling process
4. Manually review and label directories which can be used to train face recognition
5. Create a trained model (ideally which is based on an existing model)
6. Run script to output metadata about recognized faces in all images

Hopefully after all steps are performed, useful data can then be used to allow for searching and filtering the images on mikeandwan.us.

## Decisions

1. For the first attempt, we will use the 'opencv-dnn 30% confidence' detector, as it did a good job detecting faces, and was also one of the quickest.
