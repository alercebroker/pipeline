FROM python:3.9

COPY . /app
WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN pip install .[apf]

CMD ["python", "scripts/run_step.py"]
