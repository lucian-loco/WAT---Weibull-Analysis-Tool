FROM python:3.11-slim

ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown

ENV APP_GIT_COMMIT=$GIT_COMMIT \
    APP_BUILD_DATE=$BUILD_DATE

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
RUN mkdir /app/src/drawio && chmod 0777 /app/src/drawio
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./main.sh"]
