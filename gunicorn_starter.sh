#!/bin/sh
# TMP_DIR="$(mktemp -d)"
TMP_DIR="/tmp/hit-data-cache"
mkdir -p "$TMP_DIR"
export XDG_CACHE_HOME="$TMP_DIR"
export MPLCONFIGDIR="$TMP_DIR"
gunicorn --worker-tmp-dir /dev/shm --chdir app -w 2 --threads 2 -b 0.0.0.0:8888 main:app
