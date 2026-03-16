#!/bin/bash
if [ -z "$DB_USER" ] || [ -z "$DB_PASS" ] || [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ] || [ -z "$DB_SERV" ]; then
    echo "WARNING: Missing database environment variables (one of: DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_SERV)."
fi

# Default gunicorn settings
if [ -z "$GUNICORN_WORKERS" ]; then
    GUNICORN_WORKERS=1
fi

if [ -z "$GUNICORN_THREADS" ]; then
    GUNICORN_THREADS=1
fi

if [ -z "$GUNICORN_TIMEOUT" ]; then
    GUNICORN_TIMEOUT=45
fi

if [ -z "$GUNICORN_LOG_LEVEL" ]; then
    GUNICORN_LOG_LEVEL="info"
fi

if [ -z "$WEIBULL_CACHE_ENABLED" ]; then
    WEIBULL_CACHE_ENABLED="true"
fi


# TMP_DIR="$(mktemp -d)"
TMP_DIR="/tmp/hit-data-cache"
mkdir -p "$TMP_DIR"
export XDG_CACHE_HOME="$TMP_DIR"
export MPLCONFIGDIR="$TMP_DIR"
gunicorn --worker-tmp-dir /dev/shm --chdir src \
    --log-level $GUNICORN_LOG_LEVEL --workers $GUNICORN_WORKERS \
    --threads $GUNICORN_THREADS --timeout $GUNICORN_TIMEOUT \
    -b 0.0.0.0:8888 main:app
