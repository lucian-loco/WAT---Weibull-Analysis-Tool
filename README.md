# HIT Data Provider

## Introduction

This repository contains Python scripts for processing and serving data through a web server.
Additionally, it includes a Dockerfile to facilitate running the server inside a container.
The container images are stored in the registry associated with this repository.

## Endpoints

Below you will find a list of endpoints provided by the server.
Once the server is up, you can access them by opening http(s)://your-server-address/**endpoint**.
If an endpoint requires parameters, you can pass them using http(s)://your-server-address/endpoint?**param1=value1&param2=value2&param3=value3**, etc.

### weibull

Parameters:
* part - equipment code for the part to be analyzed (e.g. HCCTRI)

Returns an image presenting a Weibull probability plot for the requested part.

### crate/new

Parameters: (none)

Opens draw.io editor with a blank diagram. Preloads HIT draw.io shape libraries (from HIT project on cernbox).

### crate/edit

Parameters:
* name - name of the crate to be shown
* face - (optional) selects the side of the crate to be shown ('front' or 'back')

Opens draw.io editor with a diagram showing modules installed in a specific crate. Preloads HIT draw.io shape libraries (from HIT project on cernbox).
The diagram is drawn using data from CCDE (installed modules) and LayoutDB (modules positioning).


## Development

This project is currently deployed in the [CERN OpenShift](https://paas.cern.ch) instance (`hit-data` project) and can be accessed via https://hit-data.app.cern.ch.
Each time there is a new commit pushed to the master branch, the project will be automatically rebuilt and redeployed (with certain delay, up to 30 minutes).
Changes to both Python scripts and Dockerfile will be taken into account.

### Environment variables

This script uses database queries to fetch data. In order to connect to the database you need to set the following environment variables:
* DB_USER: user
* DB_PASS: password
* DB_HOST: connection address
* DB_PORT: connection port
* DB_SERV: service name

There is also a script which fetches draw.io template libraries from EOS. To make it work, you will need an account with read access to 'hit' EOS project
and pass the credentials using the following environmental variables:
* EOS_USER: user
* EOS_PASS: password

Draw.io editor tries to preload shape libraries from a location specified by an environmental variable:
* DRAWIO_LIBS_URL: URL providing access to the shape libraries (in form of "https://xxx/yyy"). Currently set to the public directory in HIT project cernbox.

When the project is running in an Openshift instance, you should use either 'Secrets' or 'ConfigMaps' mechanism to assign values to set these environmental variables.

### Python script

* Create a new virtual environment (`python3 -m venv venv`)
* Activate the environment (`source venv/bin/activate`)
* Install dependencies (`pip3 install -r requirements.txt`)
* Modify the code
* Start the server (`python3 src/main.py` or `./gunicorn_start.sh`)
* Navigate to http://localhost:8888/*endpoint* to test the changes.

### Dockerfile

* Modify Dockerfile
* Rebuild and run the image (`docker_start.sh`)
