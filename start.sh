#!/bin/bash

[ -f /app/config/start.sh ] && sh /app/config/start.sh

echo "ENVIRONMENT=$ENVIRONMENT" > /app/config/env.sh

exec /usr/bin/supervisord -c  /etc/supervisor/supervisord.conf
