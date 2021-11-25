FROM python:3.6-slim

RUN apt-get -y update && apt-get install -y git
RUN pip install git+https://github.com/alercebroker/db-plugins@2.0.2
WORKDIR /app
COPY . /app
CMD ["dbp", "initdb"]
