FROM python:3.11-slim

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./gunicorn_starter.sh"]
