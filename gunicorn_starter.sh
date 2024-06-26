#!/bin/sh
if [ -z "$DB_USER" ] || [ -z "$DB_PASS" ] || [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ] || [ -z "$DB_SERV" ]; then
    echo "WARNING: Missing database environment variables (one of: DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_SERV)."
fi

# TMP_DIR="$(mktemp -d)"
TMP_DIR="/tmp/hit-data-cache"
mkdir -p "$TMP_DIR"
export XDG_CACHE_HOME="$TMP_DIR"
export MPLCONFIGDIR="$TMP_DIR"
gunicorn --worker-tmp-dir /dev/shm --chdir src -w 2 --threads 2 -b 0.0.0.0:8888 main:app
