#!/bin/bash

[ -f /app/config/start.sh ] && sh /app/config/start.sh

echo "export ENVIRONMENT=$ENVIRONMENT" > /var/run/env.sh

exec /usr/bin/supervisord -c  /etc/supervisor/supervisord.conf
