FROM pytorch/torchserve:0.3.0-cpu

USER root

RUN apt-get update && apt-get upgrade -y

RUN apt-get install python3-distutils python3-dev git nano curl python3-pip libsndfile1 build-essential ffmpeg -y && \
    pip3 install -U pip && \
    pip3 install ffmpeg-python

USER model-server

RUN mkdir -p /home/model-server/services/speaker_embedding

WORKDIR /home/model-server/services/speaker_embedding

COPY speaker_embedding .

RUN git clone https://github.com/clovaai/voxceleb_trainer.git && \
    cd voxceleb_trainer && \
    git checkout 3bfd557fab5a3e6cd59d717f5029b3a20d22a281 && \
    pip3 install -r requirements.txt

RUN cd voxceleb_trainer && \
    ls && \
    patch -p1 < ../diff.patch

USER root

RUN pip install -r requirements.txt

COPY resources/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    chown -R model-server /home/model-server

USER model-server

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
