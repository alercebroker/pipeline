FROM python:3.6

RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip3 install psycopg2

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/tests
CMD apf initdb --settings_path . && python test_step.py
# CMD ["apf", "initdb", "--settings_path", "."]
# CMD ["python", "test_step.py"]