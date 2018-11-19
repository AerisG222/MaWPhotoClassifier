# mikeandwan.us Photo Classifier

A utility to try and automatically identify people in photos.  Details below are relatively unorganized notes, but hopefully will be cleaned up if I can get this working.

## Install TensorFlow Docker Container

- https://www.tensorflow.org/install/
- make sure docker is running
    - `sudo systemctl start docker.service`
- `docker pull tensorflow/tensorflow`

## Install Tensorflow Python library

```
sudo dnf install boost-devel \
                 cmake \
                 gcc-c++ \
                 graphviz \
                 lapack-devel \
                 openblas-devel \
                 pylint \
                 python3-devel \
                 python3-dlib \
                 python3-h5py \
                 python3-matplotlib \
                 python3-opencv \
                 python3-pydot \
                 python3-scipy \
                 python3-tkinter
```
- `pip3 install tensorflow --user`

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

- `pip3 install face_recognition --user`

## Install mtcnn

- `pip3 install mtcnn --user`

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
