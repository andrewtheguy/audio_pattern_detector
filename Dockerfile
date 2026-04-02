# Production Dockerfile - minimal image without ffmpeg
# ffmpeg is expected to be available on the host system if needed for non-WAV files

# --- Builder stage: compile native-helper and install all deps ---
FROM python:3.12-slim-bookworm AS builder

RUN apt-get -yqq update && \
    apt-get install -yq --no-install-recommends ca-certificates curl build-essential && \
    apt-get autoremove -y && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (needed for native-helper)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

ENV app=/usr/src/app
WORKDIR $app

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./
COPY native-helper ./native-helper

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.11,source=/uv,target=/uv \
    /uv sync --locked --no-dev --no-install-project

# Copy application code
COPY audio_pattern_detector ./audio_pattern_detector

# Install the project (builds native-helper .so)
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.11,source=/uv,target=/uv \
    /uv sync --locked --no-dev

# --- Runtime stage: minimal image with only runtime deps ---
FROM python:3.12-slim-bookworm

RUN apt-get -yqq update && \
    apt-get install -yq --no-install-recommends ca-certificates libgomp1 tini && \
    apt-get autoremove -y && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and scripts from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/audio-pattern-detector /usr/local/bin/audio-pattern-detector

# numba cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["audio-pattern-detector", "--help"]
