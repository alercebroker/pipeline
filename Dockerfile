FROM python:3.7

COPY requirements.txt /app/requirements.txt
RUN pip install numpy
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY sorting_hat_stepp /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]
