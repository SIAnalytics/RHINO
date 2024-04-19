FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /
ENV PYTHON_VERSION="3.8.5"
ENV OPENCV_VERSION="4.5.4.58"

RUN apt-get update
RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev \
        libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev \
        libbz2-dev wget tar vim libffi-dev curl make libjpeg-dev libtiff-dev \
        unzip yasm pkg-config libswscale-dev libtbb2 libtbb-dev libpng-dev \
        libavformat-dev libpq-dev libproj-dev libsqlite3-dev libexpat1-dev \
        libcurl4-gnutls-dev locales

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz \
 && tar xvf Python-$PYTHON_VERSION.tar.xz \
 && rm Python-$PYTHON_VERSION.tar.xz \
 && cd Python-$PYTHON_VERSION \
 && ./configure \
 && make -j 8 \
 && make install -j 8

RUN  cd .. \
 && rm -rf Python-$PYTHON_VERSION

RUN pip3 install --upgrade pip

RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN ln -sv /usr/local/bin/pip3 /usr/bin/pip

RUN pip3 install --no-cache-dir wheel cmake numpy scikit-image pillow yacs \
    cython matplotlib tqdm jupyter lxml fastkml shapely termcolor pyshp \
    opencv-python==$OPENCV_VERSION

RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install openmim mmengine
RUN mim install mmcv==2.0.0
RUN pip install mmdet==3.0.0

WORKDIR /mmrotate
COPY . /mmrotate
RUN pip3 install --no-cache-dir -e ".[optional]"
