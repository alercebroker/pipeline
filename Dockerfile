FROM python:3.8

ARG GH_TOKEN={GH_TOKEN}

COPY requirements.txt /app/requirements.txt
RUN pip install pandas wget validators
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

ENV NUMEXPR_MAX_THREADS=1

CMD ["python", "run_step.py"]