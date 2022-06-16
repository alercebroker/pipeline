FROM python:3.7

ARG GH_TOKEN=${GH_TOKEN}
ARG USERNAME=${USERNAME}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN git clone https://${USERNAME}:${GH_TOKEN}@github.com/alercebroker/alerce_classifiers.git
WORKDIR alerce_classifiers/
RUN pip install .["transformer_online_classifier"]

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]
