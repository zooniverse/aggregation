#!/bin/bash

[ -f /app/config/start.sh ] && sh /app/config/start.sh

exec /usr/bin/supervisord
