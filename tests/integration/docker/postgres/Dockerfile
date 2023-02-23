FROM postgres

RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y git build-essential postgresql-server-dev-all liblz4-dev \
    libssl-dev libz-dev libzstd-dev libreadline-dev
RUN git clone https://github.com/segasai/q3c.git
RUN cd q3c && make && make install
