FROM python:3.11-slim

RUN apt update && apt install -y \
    gcc \
    libsnmp-dev \
    wget \
    curl \
    xvfb \
    libasound2 \
    libgbm1

# draw.io installation
# RUN curl -s https://api.github.com/repos/jgraph/drawio-desktop/releases/latest | grep browser_download_url | grep '\.deb' | cut -d '"' -f 4 | wget -i -
# RUN apt -y install ./drawio-amd64-*.deb && rm ./drawio-amd64-*.deb
RUN wget https://github.com/jgraph/drawio-desktop/releases/download/v24.2.5/drawio-amd64-24.2.5.deb && \
    apt install -y ./drawio-amd64-24.2.5.deb && \
    rm drawio-amd64-24.2.5.deb && \
    chmod a+w -R /opt/drawio

RUN apt clean && rm -rf /var/lib/apt/lists/*

# hit-data application dependencies
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888

ENTRYPOINT ["./main.sh"]
