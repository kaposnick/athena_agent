FROM ubuntu:20.04

RUN apt-get update

RUN apt-get update

RUN apt-get install -y gir1.2-gtk-3.0

RUN apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:gnuradio/gnuradio-releases-3.9

# create user gnuario with sudo (and password gnuradio)
RUN apt-get install -y sudo
RUN useradd --create-home --shell /bin/bash -G sudo gnuradio
RUN echo 'gnuradio:gnuradio' | chpasswd

# I create a dir at home which I'll use to persist after the container is closed (need to change it's ownership)
RUN mkdir /home/gnuradio/persistent  && chown gnuradio /home/gnuradio/persistent

RUN apt-get update

RUN apt-get install -y gnuradio

# installing other packages needed for downloading and installing OOT modules
RUN apt-get install -y gnuradio-dev cmake git libboost-all-dev libcppunit-dev liblog4cpp5-dev python3-pygccxml pybind11-dev liborc-dev

# of course, nothing useful can be done without vim
RUN apt-get install -y vim 

USER root

WORKDIR /home/gnuradio

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3/dist-packages"

COPY gnuradio/testbed_wireless_channel.py /home/gnuradio
