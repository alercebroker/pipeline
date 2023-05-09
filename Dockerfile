FROM python:3.10

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
WORKDIR /app
COPY ./scripts /app/scripts
COPY ./schemas /app/schemas
COPY ./simulator /app/simulator
COPY ./settings.py /app/settings.py

CMD ["python", "scripts/run_step.py"]
