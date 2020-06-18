FROM python:3.6

RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip3 install psycopg2

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]