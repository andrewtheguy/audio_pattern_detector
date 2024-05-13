FROM python:3.12-slim-bookworm as builder

# Copy the necessary files for Poetry to the generate the requirements file from
COPY ../pyproject.toml ../poetry.lock /tmp/

RUN pip3 install poetry==1.8.* poetry-plugin-export

RUN cd /tmp && poetry export --without-hashes -f requirements.txt -o requirements.txt


FROM rclone/rclone:1.66 as rclone

FROM python:3.12-slim-bookworm

RUN     apt-get -yqq update && \
        apt-get install -yq --no-install-recommends ca-certificates expat libgomp1 tini git && \
        apt-get autoremove -y && \
        apt-get clean -y && rm -rf /var/lib/apt/lists/*


COPY --from=builder /tmp/requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

COPY --from=andrewchen5678/static-ffmpeg:20211210-3 /ffmpeg /usr/local/bin/
COPY --from=andrewchen5678/static-ffmpeg:20211210-3 /ffprobe /usr/local/bin/

#VOLUME '/usr/src/app'

ENV app /usr/src/app
WORKDIR $app

COPY . /usr/src/app

#COPY --from=rclone /usr/local/bin/rclone /usr/local/bin/rclone

#COPY --chown=1000:1000 --from=intermediate /tmp/frontend/public /usr/src/app/public

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python", "-u", "schedule.py"]