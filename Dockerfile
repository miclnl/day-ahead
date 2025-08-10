ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG BUILD_ARCH
ENV ENV_BUILD_ARCH=$BUILD_ARCH

# Note: Using root user for Home Assistant compatibility
# RUN useradd --system --no-create-home --shell /sbin/nologin --home-dir /app daouser

# Add development tools needed for Python packages with C extensions
RUN apt-get update && apt-get install -y \
    # Python and development tools
    python3 python3-pip python3-dev python3-venv \
    # Database clients and dev headers
    mariadb-client libmariadb-dev postgresql-client sqlite3 \
    # Compilation tools
    gcc g++ libc6-dev linux-libc-dev build-essential cmake \
    # Graphics and image processing
    libjpeg-dev libpng-dev libfreetype6-dev \
    # Math libraries for numpy/scipy
    libopenblas-dev liblapack-dev \
    # Crypto libraries
    libffi-dev libssl-dev \
    # Other common dependencies  
    gfortran pkg-config \
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

#benodigde libraries voor mip (some may not exist in Alpine)
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

# Python is already installed in debian-base image
# Just ensure pip and venv are available
RUN python3 -m pip install --break-system-packages --upgrade pip

ENV VIRTUAL_ENV=/app/dao/venv/day_ahead
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/
RUN $VIRTUAL_ENV/bin/pip install uv
# Split requirements for better caching
COPY requirements-base.txt /tmp/
RUN $VIRTUAL_ENV/bin/uv pip install --prerelease=allow -r /tmp/requirements-base.txt

# Install PyTorch CPU-only separately (large, cacheable layer)
RUN $VIRTUAL_ENV/bin/uv pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.0.0,\<2.3.0 torchvision>=0.15.0,\<0.18.0

# Install remaining requirements
RUN $VIRTUAL_ENV/bin/uv pip install --prerelease=allow -r /tmp/requirements.txt
COPY run/ /app/dao/run/

# Using root user for Home Assistant compatibility
EXPOSE 5001
WORKDIR /app/dao/prog
ENTRYPOINT ["python3", "da_entrypoint.py"]

# Labels
LABEL \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION}

