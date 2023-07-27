FROM python:3.7

EXPOSE 5000

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r ./requirements.txt && pip install gunicorn

WORKDIR /app/service

CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "8", "stamp_clf_service:app"]
