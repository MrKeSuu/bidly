FROM python:3.9

WORKDIR /home/user

RUN apt-get -y update && apt-get install -y libboost-system-dev libboost-filesystem-dev \
    libboost-chrono-dev libboost-program-options-dev libboost-test-dev libboost-thread-dev \
    libboost-iostreams-dev \
    vim less

RUN mkdir -p install/dds/
COPY install/dds/ install/dds/
RUN cd install/dds/src && make && cd ~

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN cp install/dds/src/libdds.so pythondds_min/

ENTRYPOINT ["make"]
CMD ["test"]
