#!/bin/sh
export ELECTRON_DISABLE_SECURITY_WARNINGS="true"
export ELECTRON_ENABLE_LOGGING="false"
export DISPLAY=":99"

Xvfb $DISPLAY -screen 0 1024x768x16 -nolisten unix &
./update_drawio.sh
./gunicorn_starter.sh
