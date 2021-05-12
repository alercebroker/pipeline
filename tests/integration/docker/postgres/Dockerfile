FROM postgres

RUN apt-get update -y
RUN apt-get install -y git build-essential postgresql-server-dev-13 libssl-dev libz-dev libreadline-dev
RUN git clone https://github.com/segasai/q3c.git
RUN cd q3c && make && make install
