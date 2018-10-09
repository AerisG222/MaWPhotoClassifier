# mikeandwan.us Photo Classifier

A utility to try and automatically identify people in photos.  Details below are relatively unorganized notes, but hopefully will be cleaned up if I can get this working.

## Install TensorFlow Docker Container

- https://www.tensorflow.org/install/
- make sure docker is running
    - `sudo systemctl start docker.service`
- `docker pull tensorflow/tensorflow`

## Install Tensorflow Python library

- `sudo dnf install cmake python3-devel python3-tkinter python3-dlib boost-devel lapack-devel openblas-devel pylint`
- `pip3 install tensorflow --user`
- `pip3 install matplotlib --user`

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

## Build DLIB (might not be required)

- `sudo dnf install python3-devel openblas-devel lapack-devel`
- `git clone https://github.com/davisking/dlib.git`

## Install face_recognition

- `pip3 install face_recognition --user`

## Install SciKitLearn

- `pip3 install scikit-learn --user`
