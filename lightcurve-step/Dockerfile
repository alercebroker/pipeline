FROM python:3.9

RUN pip install poetry

WORKDIR /app

COPY poetry.lock pyproject.toml /app/
RUN poetry install --no-interaction

COPY . /app

CMD ["poetry", "run", "python", "scripts/run_step.py"]
