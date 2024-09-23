FROM python:3.11-slim

RUN apt update && apt install -y wget libsnmp-dev gcc

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./main.sh"]
