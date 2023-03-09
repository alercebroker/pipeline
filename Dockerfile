FROM python:3.10

RUN pip install poetry
RUN poetry install --with dev

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["poetry", "run", "python", "run_step.py"]
