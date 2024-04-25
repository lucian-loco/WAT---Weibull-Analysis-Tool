#!/bin/sh
gunicorn  --worker-tmp-dir /dev/shm --chdir app -w 2 --threads 2 -b 0.0.0.0:8888 main:app
