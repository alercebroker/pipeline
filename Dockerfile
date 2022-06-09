FROM python:3.6

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install python-snappy
RUN apt update -y
RUN apt install libsnappy-dev -y
WORKDIR /app
COPY ./scripts /app/scripts
COPY ./schemas /app/schemas
COPY ./simulator /app/simulator
COPY ./settings.py /app/settings.py


WORKDIR /app/scripts

CMD ["bash", "entrypoint.sh"]
