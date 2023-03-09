FROM python:3.9

RUN pip install poetry

COPY . /app
WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN poetry install --with dev

WORKDIR /app/scripts

CMD ["poetry", "run", "python", "run_step.py"]
