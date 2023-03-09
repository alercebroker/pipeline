FROM python:3.9

COPY . /app
WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN pip install .

WORKDIR /app/scripts

CMD ["python", "run_step.py"]
