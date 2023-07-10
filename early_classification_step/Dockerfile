FROM python:3.7

#Model installation
COPY model.zip /app/model.zip
RUN unzip /app/model.zip -d /app/model
RUN pip install -r /app/model/requirements.txt

#Code Installation
RUN apt-get update && apt-get install -y libpq-dev gcc
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]
