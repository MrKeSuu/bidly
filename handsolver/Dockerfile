FROM python:3.9

WORKDIR /home/user

RUN mkdir -p /home/user/resources

RUN apt-get -y update && apt-get install -y libboost-system-dev libboost-filesystem-dev \
    libboost-chrono-dev libboost-program-options-dev libboost-test-dev libboost-thread-dev \
    libboost-iostreams-dev
COPY install/dds/ resources/dds/
RUN cd resources/dds/src && make && cd ~

COPY install/requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /home/user/code
COPY pythondds_min code/pythondds_min/
COPY Makefile .
RUN cp resources/dds/src/libdds.so code/pythondds_min/

ENTRYPOINT ["make"]
CMD ["test"]
