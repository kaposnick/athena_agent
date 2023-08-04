FROM tensorflow/tensorflow

COPY ./scheduler /scheduler
WORKDIR /scheduler
