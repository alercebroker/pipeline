FROM python:3.8

COPY requirements.txt /app/requirements.txt

RUN pip install --use-deprecated=legacy-resolver -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

ENV NUMEXPR_MAX_THREADS=1

CMD ["python", "run_step.py"]
