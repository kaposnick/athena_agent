FROM ubuntu:22.04

RUN apt-get update -y && \
    apt-get install -y software-properties-common

RUN apt-get update -y && \
	apt-get install -y cmake \
	libfftw3-dev \
	libmbedtls-dev \
	libboost-program-options-dev \
	libconfig++-dev \
	libsctp-dev \
	libzmq3-dev

RUN apt-get install -y build-essential

COPY ./srsRAN /srsRAN
WORKDIR srsRAN/build
RUN cmake ../
RUN make -j 4 && make install && ldconfig
RUN ./srsran_install_configs.sh user
WORKDIR /root

RUN apt-get install iproute2 -y
