# CUDA 11をベースにしたUbuntu 20.04イメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG user

# 必要なパッケージのインストール
RUN apt-get update 
RUN apt-get install -y python3.10 python3-pip git libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg wget zip xvfb x11-apps x11-xserver-utils


# WORKDIR /home/${user}/octo
# COPY . .

# RUN \
#   pip3 install -e . && \
#   pip3 install -r requirements.txt && \
#   pip3 install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
