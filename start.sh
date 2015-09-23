#!/bin/bash

[ -x /app/config/start.sh ] && /app/config/start.sh

exec /usr/bin/supervisord
