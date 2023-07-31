FROM postgres

RUN apt-get update -y
RUN echo en_US.UTF-8 UTF-8 > /etc/locale.gen
RUN locale-gen en_US.UTF-8
RUN apt-get install -y git gcc postgresql-server-dev-all liblz4-dev \
    libz-dev libzstd-dev libreadline-dev
RUN git clone https://github.com/segasai/q3c.git
RUN cd q3c && make && make install
