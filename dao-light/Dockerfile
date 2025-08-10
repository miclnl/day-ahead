ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG BUILD_ARCH
ENV ENV_BUILD_ARCH=$BUILD_ARCH

# Note: Using root user for Home Assistant compatibility
# RUN useradd --system --no-create-home --shell /sbin/nologin --home-dir /app daouser

# Debian base - minimal packages for Light version
RUN apt-get update && apt-get install -y \
    # Python and pip
    python3 python3-pip python3-dev python3-venv \
    # Database clients and minimal dev headers
    mariadb-client libmariadb-dev postgresql-client sqlite3 \
    # Minimal compilation tools
    gcc libc6-dev linux-libc-dev build-essential \
    # Essential graphics support
    libjpeg-dev \
    # Crypto support
    libffi-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Copy software for add-on (using root for HA compatibility)  
RUN mkdir -p /app/dao

# Copy static configuration first (rarely changes)
COPY data /app/daodata/
COPY __init__.py /app/dao/

# Copy application code later (changes more often)
COPY prog /app/dao/prog/
COPY webserver /app/dao/webserver/

#version
ARG BUILD_VERSION
ENV DAO_VERSION=$BUILD_VERSION
RUN printf  '__version__ ="%s"\n' "$DAO_VERSION" > /app/dao/prog/version.py

#benodigde libraries voor mip (minimal for light version)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/* || true

COPY miplib.tar.gz /tmp/
WORKDIR /app/dao/prog
ENV BUILD_ARCH="$ENV_BUILD_ARCH"
ENV PYTHONPATH="/app:/app/dao:/app/dao/prog"

RUN if [ "${BUILD_ARCH}" = "aarch64" ]; then \
      tar -xvf /tmp/miplib.tar.gz -C /app/dao/prog \
   ; fi

# Set environment variables for runtime
ENV PMIP_CBC_LIBRARY="/app/dao/prog/miplib/lib/libCbc.so"
ENV LD_LIBRARY_PATH="/app/dao/prog/miplib/lib/"

# Python already added above, just ensure pip is upgraded
RUN python3 -m pip install --break-system-packages --upgrade pip

# Light version: install with optimized layers for caching
COPY requirements-base.txt /tmp/
RUN pip3 install --break-system-packages -r /tmp/requirements-base.txt

# Install remaining requirements in separate layer
COPY requirements.txt /tmp/
RUN pip3 install --break-system-packages -r /tmp/requirements.txt
COPY run/ /app/dao/run/

# Using root user for Home Assistant compatibility
EXPOSE 5002
WORKDIR /app/dao/prog
ENTRYPOINT ["python3", "da_entrypoint_light.py"]

# Labels
LABEL \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION}

