FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN uv sync --locked --no-dev

# numba patch
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python","-u","match.py"]