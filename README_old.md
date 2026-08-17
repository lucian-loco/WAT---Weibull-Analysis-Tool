# WAT - Weibull Analysis Tool

## Introduction

This repository contains Python scripts for Weibull analysis through a web server as a tool.
Additionally, it includes a Dockerfile to facilitate running the server inside a container.
The container images are stored in the registry associated with this repository.

## Endpoint

Below you will find the endpoint provided by the server.
Once the server is up, you can access them by opening http(s)://your-server-address/**endpoint**.
If an endpoint requires parameters, you can pass them using http(s)://your-server-address/endpoint?**param1=value1&param2=value2&param3=value3**, etc.

### Weibull analysis

Parameters:
* part - equipment code for the part to be analyzed (e.g. HCCTRI)

Returns an image presenting a Weibull probability plot for the requested part.


### Environment variables

This script uses database queries to fetch data. In order to connect to the database you need to set the following environment variables:
* DB_USER: user
* DB_PASS: password
* DB_HOST: connection address
* DB_PORT: connection port
* DB_SERV: service name

When the project is running in an Openshift instance, you should use either 'Secrets' or 'ConfigMaps' mechanism to assign values to set these environmental variables.

### Python script

* Create a new virtual environment (`python3 -m venv venv`)
* Activate the environment (`source venv/bin/activate`)
* Install dependencies (`pip3 install -r requirements.txt`)
* Modify the code
* Start the server (`python3 src/main.py`)
* Navigate to http://localhost:8888/*endpoint* to test the changes.

### Dockerfile

* Modify Dockerfile
* Rebuild and run the image (`docker_start.sh`)
