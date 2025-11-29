# Production Dockerfile - minimal image without ffmpeg
# ffmpeg is expected to be available on the host system if needed for non-WAV files
FROM python:3.12-slim-bookworm

RUN apt-get -yqq update && \
    apt-get install -yq --no-install-recommends ca-certificates libgomp1 tini && \
    apt-get autoremove -y && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV app=/usr/src/app
WORKDIR $app

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.11,source=/uv,target=/uv \
    /uv sync --locked --no-dev --no-install-project

# Copy application code
COPY audio_pattern_detector ./audio_pattern_detector

# Install the project
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.11,source=/uv,target=/uv \
    /uv sync --locked --no-dev

# numba cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["audio-pattern-detector", "--help"]
