FROM python:3.12-slim-bookworm

RUN     apt-get -yqq update && \
        apt-get install -yq --no-install-recommends ca-certificates expat libgomp1 tini git && \
        apt-get autoremove -y && \
        apt-get clean -y && rm -rf /var/lib/apt/lists/*


COPY --from=mwader/static-ffmpeg:7.0-1 /ffmpeg /usr/local/bin/
COPY --from=mwader/static-ffmpeg:7.0-1 /ffprobe /usr/local/bin/

#VOLUME '/usr/src/app'

ENV app=/usr/src/app
WORKDIR $app

COPY . /usr/src/app

ENV UV_SYSTEM_PYTHON=1
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.11,source=/uv,target=/bin/uv \
    uv sync --locked --no-dev

# numba patch
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python","-u","match.py"]