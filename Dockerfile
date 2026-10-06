# Build stage
FROM rust:1.91-slim-trixie AS builder
ARG TARGETARCH

WORKDIR /build

COPY . .

# Build the release binary with architecture-specific cache mounts
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-v2-${TARGETARCH} \
    --mount=type=cache,target=/build/target,id=cargo-target-v2-${TARGETARCH} \
    cargo build --release --locked && \
    cp target/release/audio-pattern-detector /audio-pattern-detector

# Runtime stage - minimal image without ffmpeg (builds from source).
# WAV input is decoded natively; ffmpeg is only needed for other formats and is
# expected to be provided by the consuming image.
FROM debian:trixie-slim AS runtime

LABEL org.opencontainers.image.source=https://github.com/andrewtheguy/audio_pattern_detector

RUN apt-get update && apt-get install -y \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /audio-pattern-detector /usr/local/bin/audio-pattern-detector

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["audio-pattern-detector", "--help"]

# Runtime stage for pre-built binary (used by CI to avoid double build)
FROM debian:trixie-slim AS runtime-prebuilt

LABEL org.opencontainers.image.source=https://github.com/andrewtheguy/audio_pattern_detector

RUN apt-get update && apt-get install -y \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Binary must be passed via build context
COPY audio-pattern-detector /usr/local/bin/audio-pattern-detector

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["audio-pattern-detector", "--help"]
