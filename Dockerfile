FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]