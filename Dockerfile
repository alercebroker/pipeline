FROM python:3.6

COPY requirements.txt /app/requirements.txt
RUN pip install numpy Cython
RUN pip install -r /app/requirements.txt
COPY late_classifier /late_classifier
COPY turbo-fats /turbo-fats
RUN pip install -e /late_classifier
RUN pip install -e /turbo-fats

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]