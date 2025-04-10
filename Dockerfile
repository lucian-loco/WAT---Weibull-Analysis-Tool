FROM python:3.11-slim

RUN apt update && apt install -y wget libsnmp-dev gcc curl

RUN curl -s https://api.github.com/repos/jgraph/drawio-desktop/releases/latest | grep browser_download_url | grep '\.deb' | cut -d '"' -f 4 | wget -i -
RUN apt -y install ./drawio-amd64-*.deb

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./main.sh"]
