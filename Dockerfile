# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition

FROM nvidia/cuda:9.0-base

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-h5py \
    python3-matplotlib \
    python3-numpy \
    python3-opencv \
    python3-pydot \
    python3-scipy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN mkdir /test

COPY ./tensorflow_install_test.py /test

RUN cd /test && \
    pip3 install tensorflow-gpu

CMD cd /test && \
    python3 tensorflow_install_test.py
