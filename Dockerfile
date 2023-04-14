FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-opencv \
    unzip \
    sudo \
    vim

RUN pip install tensorboardX \
    gdown pycocotools pipenv \
    ptflops wget pandas \
    pycocotools numpy opencv-python \
    tqdm tensorboard tensorboardX \
    pyyaml webcolors jsonlines \
    gradio seaborn gevent gunicorn flask \
    wandb pyyaml webcolors tensorboard matplotlib \
    scipy scikit-learn jupyter


RUN pip install torch_tb_profiler compressai
RUN pip install h5py

RUN pip install "numpy==1.23.5"

ARG UID
ARG GID
ARG USER
ARG GROUP
RUN groupadd -g $GID $GROUP
RUN useradd -r -s /bin/false -g $GROUP -G sudo -u $UID $USER
RUN mkdir /home/$USER
RUN chmod -R 777 /home/$USER

CMD /bin/bash


#COPY requirements.txt ./requirements.txt
#RUN pip install -r requirements.txt
