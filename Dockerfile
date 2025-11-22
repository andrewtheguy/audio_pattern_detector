FROM python:3.12-slim-bookworm AS builder

# Install git for resolving git dependencies
RUN apt-get -yqq update && \
    apt-get install -yq --no-install-recommends git && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Copy the necessary files for uv to generate the requirements file from
COPY pyproject.toml uv.lock /tmp/

RUN pip3 install uv

WORKDIR /tmp
RUN uv pip compile pyproject.toml -o requirements.txt


FROM python:3.12-slim-bookworm

RUN     apt-get -yqq update && \
        apt-get install -yq --no-install-recommends ca-certificates expat libgomp1 tini git && \
        apt-get autoremove -y && \
        apt-get clean -y && rm -rf /var/lib/apt/lists/*


COPY --from=builder /tmp/requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY --from=mwader/static-ffmpeg:7.0-1 /ffmpeg /usr/local/bin/
COPY --from=mwader/static-ffmpeg:7.0-1 /ffprobe /usr/local/bin/

#VOLUME '/usr/src/app'

ENV app=/usr/src/app
WORKDIR $app

COPY . /usr/src/app

# numba patch
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python","-u","match.py"]