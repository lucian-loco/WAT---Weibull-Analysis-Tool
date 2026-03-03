FROM python:3.11-slim

RUN apt update && apt install -y \
    gcc \
    libsnmp-dev \
    wget \
    curl

RUN apt clean && rm -rf /var/lib/apt/lists/*

# hit-data application dependencies
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./main.sh"]
