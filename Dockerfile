# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition

FROM nvidia/cuda:9.0-base

RUN apt-get update && \
    apt-get -y install python3 python3-pip

RUN mkdir /test

COPY ./tensorflow_install_test.py /test

RUN cd /test && \
    pip3 install tensorflow-gpu

CMD cd /test && \
    python3 tensorflow_install_test.py
