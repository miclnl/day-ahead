#!/usr/bin/env bashio

# exit immediately if a command exits with a non-zero status
# set -e

dir="/config/dao_data"
if [ ! -d "$dir" ]; then
  bashio::log.info "=> directory dao_data made, files copied"
  cp -r /tmp/daodata /config/dao_data
  file=/config/dao_data/options.json
  if [ ! -L "$file" ]; then
    cp /config/dao_data/options_start.json $file
  fi
  file=/config/dao_data/secrets.json
  if [ ! -L "$file" ]; then
    cp /config/dao_data/secrets_vb.json $file
  fi
else
  bashio::log.info "=> directory dao_data exist"
fi

cd /root/dao/prog
file=../data
if [ -L "$file" ]
then
  bashio::log.info "=> /root/dao/data exist"
else
  bashio::log.info "=> /root/dao/data doesn't exist, made"
  ln -s /config/dao_data $file
fi

# Older versions linked the data directory into the web server's static
# folder, which made secrets.json and the database downloadable. Remove the
# link if it is still there; the dashboard reads ../data directly now.
cd /root/dao/webserver/
if [ -L app/static/data ]; then
  bashio::log.info "=> removing obsolete link /root/dao/webserver/app/static/data"
  rm -f app/static/data
fi

# The dashboard only answers requests that come through Home Assistant's
# ingress unless the operator opts in to direct access on the mapped port.
if bashio::config.true 'allow_direct_access'; then
  bashio::log.warning "Dashboard is bereikbaar zonder Home Assistant login (allow_direct_access)"
  export DAO_ALLOW_DIRECT=1
fi

export PYTHONPATH="/root:/root/dao:/root/dao/lib:/root/dao/prog"
cd /root/dao/prog
# A failure here is not fatal -- the add-on still starts and the optimiser
# still runs -- but it does mean a schema migration or an index/table
# creation did not happen, which then shows up much later as a confusing
# error during a task. Log it as a warning so it is visible in the add-on
# log instead of blending into the info lines.
python3 check_db.py || bashio::log.warning \
  "check_db.py is mislukt; database-migraties zijn mogelijk niet uitgevoerd"

if [ -d /config/miplib/lib ]; then
  bashio::log.info "Copying saved miplib-binaries"
  cp -a /config/miplib miplib
elif bashio::config.true 'use_self_compiled_miplib'; then
  bashio::log.info "Building new miplib-binaries"
  chmod a+x build_miplib.sh
  ./build_miplib.sh
fi

if [ -d miplib/lib ]; then
  export PMIP_CBC_LIBRARY="/root/dao/prog/miplib/lib/libCbc.so"
  export LD_LIBRARY_PATH="/root/dao/prog/miplib/lib"
  echo 'export PMIP_CBC_LIBRARY="/root/dao/prog/miplib/lib/libCbc.so"' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH="/root/dao/prog/miplib/lib/"' >> ~/.bashrc
fi

cd /root/dao/prog
exec bash ./watchdog.sh
