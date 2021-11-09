FROM python:3.9


WORKDIR /home/user

RUN mkdir -p /home/user/resources

RUN apt-get -y update && apt-get install -y libboost-system-dev libboost-filesystem-dev \
    libboost-chrono-dev libboost-program-options-dev libboost-test-dev libboost-thread-dev \
    libboost-iostreams-dev

COPY install/dds/ resources/dds/
RUN cd resources/dds/src && make

CMD ["ls", "-l", "/home/user/resources/dds/src/libdds.so"]
