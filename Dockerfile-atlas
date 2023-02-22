FROM python:3.8
ARG GITHUB_TOKEN

RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

CMD ["python", "run_step.py"]