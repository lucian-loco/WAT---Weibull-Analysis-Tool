FROM python:3.11-slim

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

RUN apt update && apt install -y wget

COPY . /app
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./main.sh"]
