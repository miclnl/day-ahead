#!/bin/bash
dir="/config/dao_light_data"
if [ ! -d "$dir" ]; then
  echo "=> directory dao_light_data made, files copied"
  cp -r /tmp/daodata /config/dao_light_data
  file=/config/dao_light_data/options.json
  if [ ! -L "$file" ]; then
    cp /config/dao_light_data/options_start.json $file
  fi
  file=/config/dao_light_data/secrets.json
  if [ ! -L "$file" ]; then
    cp /config/dao_light_data/secrets_vb.json $file
  fi
else
  echo "=> directory dao_light_data exist"
fi

cd /root/dao/prog
file=../data
if [ -L "$file" ]
then
  echo "=> /root/dao/data exist"
else
  echo "=> /root/dao/data doesn't exist, made"
  ln -s /config/dao_light_data $file
fi

cd /root/dao/webserver/
file=app/static/data
if [ -L "$file" ]
then
  echo "=> /root/dao/webserver/app/static/data exist"
else
  echo "=> /root/dao/webserver/app/static/data doesn't exist, made"
  ln -s /config/dao_light_data $file
fi

dir="/root/dao/prog/miplib/"
if [ -d "$dir" ]; then
  export PMIP_CBC_LIBRARY="/root/dao/prog/miplib/lib/libCbc.so"
fi

export PYTHONPATH="/root:/root/dao:/root/dao/prog"
cd /root/dao/prog
python3 check_db.py

cd /root/dao/webserver/
gunicorn --config gunicorn_config.py app:app &

cd /root/dao/prog

echo "=> Starting DAO Light - Simple and Stable"

# NO smart services in light mode - keep it simple
echo "=> Light mode: Smart services disabled for stability"

# Use simple scheduler (no APScheduler, no modern features)
if [ -f "da_simple_scheduler.py" ]; then
  echo "=> Starting simple thread-based scheduler (Light Mode)"
  bash ./watchdog.sh python3 da_simple_scheduler.py
else
  echo "=> Fallback to original scheduler"  
  bash ./watchdog.sh python3 da_scheduler.py
fi