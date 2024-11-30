FROM python:3.12-slim-bookworm as builder

# Copy the necessary files for Poetry to the generate the requirements file from
COPY ../pyproject.toml ../poetry.lock /tmp/

RUN pip3 install poetry==1.8.* poetry-plugin-export

RUN cd /tmp && poetry export --without test --without-hashes -f requirements.txt -o requirements.txt


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

ENV app /usr/src/app
WORKDIR $app

COPY . /usr/src/app

# numba patch
ENV NUMBA_CACHE_DIR=/tmp/numba_cache_dir

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python","-u","match.py"]