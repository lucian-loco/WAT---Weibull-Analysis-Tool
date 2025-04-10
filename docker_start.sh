#!/bin/sh
# Script to run the Docker image locally
# Note that it requires several environment variables (DB_*) in order to access the database
# and an account with access to 'hit' EOS project to update draw.io templates (EOS_*)
docker build -t hit-data . && \
docker run --rm -p 8888:8888 -it \
    -e EOS_USER="$EOS_USER" \
    -e EOS_PASS="$EOS_PASS" \
    -e DB_USER="$DB_USER" \
    -e DB_PASS="$DB_PASS" \
    -e DB_HOST="$DB_HOST" \
    -e DB_PORT="$DB_PORT" \
    -e DB_SERV="$DB_SERV" \
    --entrypoint /bin/bash \
    hit-data
