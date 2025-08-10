ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG BUILD_ARCH
ENV ENV_BUILD_ARCH=$BUILD_ARCH

# Create non-root user for security
RUN useradd --system --no-create-home --shell /sbin/nologin --home-dir /app daouser

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

# Copy software for add-on to non-root location
RUN mkdir -p /app/dao && chown -R daouser:daouser /app
COPY --chown=daouser:daouser __init__.py /app/dao/
COPY --chown=daouser:daouser prog /app/dao/prog/
COPY --chown=daouser:daouser webserver /app/dao/webserver/
COPY --chown=daouser:daouser data /app/daodata/

#version
ARG BUILD_VERSION
ENV DAO_VERSION=$BUILD_VERSION
RUN printf  '__version__ ="%s"\n' "$DAO_VERSION" > /app/dao/prog/version.py

#benodigde libraries voor mip (some may not exist in Alpine)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/* || true

COPY --chown=daouser:daouser miplib.tar.gz /tmp/
WORKDIR /app/dao/prog
ENV BUILD_ARCH="$ENV_BUILD_ARCH"
ENV PYTHONPATH="/app:/app/dao:/app/dao/prog"

RUN if [ "${BUILD_ARCH}" = "aarch64" ]; then \
      tar -xvf /tmp/miplib.tar.gz -C /app/dao/prog && \
      chown -R daouser:daouser /app/dao/prog/miplib \
   ; fi

# Set environment variables for runtime
ENV PMIP_CBC_LIBRARY="/app/dao/prog/miplib/lib/libCbc.so"
ENV LD_LIBRARY_PATH="/app/dao/prog/miplib/lib/"

# Python is already installed in debian-base image
# Just ensure pip and venv are available
RUN python3 -m pip install --break-system-packages --upgrade pip

ENV VIRTUAL_ENV=/app/dao/venv/day_ahead
RUN python3 -m venv $VIRTUAL_ENV && chown -R daouser:daouser $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip3 install --break-system-packages uv
RUN uv pip install --prerelease=allow -r /tmp/requirements.txt
COPY --chown=daouser:daouser run /app/dao/prog/

# Switch to non-root user
USER daouser
EXPOSE 5001
WORKDIR /app/dao/prog
ENTRYPOINT ["python3", "da_entrypoint.py"]

# Labels
LABEL \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION}

