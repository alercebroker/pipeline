FROM python:3.8

ARG GH_TOKEN={GH_TOKEN}

COPY requirements.txt /app/requirements.txt

RUN pip install --use-deprecated=legacy-resolver -r /app/requirements.txt

WORKDIR /app
COPY . /app

ENV NUMEXPR_MAX_THREADS=1

CMD ["python", "scripts/run_step.py"]
