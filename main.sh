#!/bin/sh
export ELECTRON_DISABLE_SECURITY_WARNINGS="true"
export ELECTRON_ENABLE_LOGGING="false"
export DISPLAY=":99"

./update_drawio.sh
./gunicorn_starter.sh
