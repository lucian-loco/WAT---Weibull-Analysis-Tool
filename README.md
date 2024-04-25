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
