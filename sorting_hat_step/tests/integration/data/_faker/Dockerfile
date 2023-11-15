FROM python:3.10-slim

RUN pip install confluent-kafka fastavro
COPY . .
ENTRYPOINT [ "python", "public.py" ]