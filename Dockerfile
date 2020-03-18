FROM python:3.6
COPY requirements.txt /app/requirements.txt
COPY /APF /APF 
COPY /late_classifier /late_classifier
COPY /paps /paps
COPY /turbo-fats /turbo-fats
RUN pip install -e /APF
RUN pip install -e /late_classifier
RUN pip install Cython
RUN pip install -e /paps
RUN pip install -e /turbo-fats
RUN pip install -r /app/requirements.txt
RUN pip install psycopg2

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]