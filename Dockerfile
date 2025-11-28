FROM python:3.12-slim-bookworm

# Build argument to include dev dependencies (for testing)
ARG INCLUDE_DEV=false

RUN     apt-get -yqq update && \
        apt-get install -yq --no-install-recommends ca-certificates expat libgomp1 tini git && \
        apt-get autoremove -y && \
        apt-get clean -y && rm -rf /var/lib/apt/lists/*


COPY --from=mwader/static-ffmpeg:7.0-1 /ffmpeg /usr/local/bin/
COPY --from=mwader/static-ffmpeg:7.0-1 /ffprobe /usr/local/bin/

# Copy uv for runtime use (testing)
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /usr/local/bin/uv

#VOLUME '/usr/src/app'

ENV app=/usr/src/app
WORKDIR $app

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN if [ "$INCLUDE_DEV" = "true" ]; then \
        uv sync --locked --no-install-project; \
    else \
        uv sync --locked --no-dev --no-install-project; \
    fi

# Copy application code after dependencies are installed
COPY . ./

# numba patch
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python","-u","match.py"]